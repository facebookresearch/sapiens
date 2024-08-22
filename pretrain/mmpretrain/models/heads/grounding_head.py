# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.models.utils.box_utils import (box_cxcywh_to_xyxy,
                                               generalized_box_iou)
from mmpretrain.registry import MODELS, TOKENIZER


@MODELS.register_module()
class GroundingHead(BaseModule):
    """bbox Coordination generation head for multi-modal pre-trained task,
    adapted by BLIP. Normally used for visual grounding.

    Args:
        loss: dict,
        decoder: dict,
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        decoder: dict = None,
        tokenizer: dict = None,
        box_l1_loss_coeff=4.0,
        box_giou_loss_coeff=2.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super(GroundingHead, self).__init__(init_cfg=init_cfg)
        ''' init the decoder from med_config'''
        self.decoder = None
        if decoder:
            self.decoder = MODELS.build(decoder)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=-100)

        self.box_l1_loss_coeff = box_l1_loss_coeff
        self.box_giou_loss_coeff = box_giou_loss_coeff

        if isinstance(tokenizer, dict):
            self.tokenizer = TOKENIZER.build(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.image_res = 640
        prefix_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(['[unused339]']))
        target_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(
                [f'[unused{340+_}]' for _ in range(self.image_res + 1)]))
        self.register_buffer('prefix_ids', prefix_ids)
        self.register_buffer('target_ids', target_ids)

        bbox_prob_mask = torch.zeros(len(self.tokenizer))
        bbox_prob_mask[self.target_ids[0]:self.target_ids[-1] + 1] = 1
        bbox_prob_mask = (1.0 - bbox_prob_mask) * -10000.0
        self.register_buffer('bbox_prob_mask', bbox_prob_mask)
        self.bin_start_idx = self.target_ids[0]

    def forward(self, text_embedding, text_embedding_mask,
                encoder_hidden_states, encoder_attention_mask):

        # localize prompt token, text embedding

        merged_encode_hs = torch.cat([encoder_hidden_states, text_embedding],
                                     1)
        merge_att_mask = torch.cat(
            [encoder_attention_mask, text_embedding_mask], 1)

        loc_prompt = self.prompt.weight.T
        loc_prompt = torch.repeat_interleave(loc_prompt,
                                             merge_att_mask.shape[0],
                                             0).unsqueeze(1)

        loc_prompt_mask = torch.ones(loc_prompt.shape[:-1]).long().to(
            loc_prompt.device)

        decoder_out = self.decoder(
            inputs_embeds=loc_prompt,
            attention_mask=loc_prompt_mask,
            encoder_hidden_states=merged_encode_hs,
            encoder_attention_mask=merge_att_mask,
            output_hidden_states=True,
            labels=None,
        )
        decoder_hs = decoder_out.hidden_states[-1][:, 0, :]
        box_pred = self.box_head(decoder_hs)
        return decoder_out, decoder_hs, box_pred

    def loss(self,
             text_embedding,
             text_embedding_mask,
             encoder_hidden_states,
             encoder_attention_mask,
             decoder_targets,
             return_scores=False):
        """Calculate losses from the extracted features.

        Args:
            feats (dict): The features extracted from the backbone.
            data_samples (List[BaseDataElement]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        merged_encode_hs = torch.cat([encoder_hidden_states, text_embedding],
                                     1)
        merge_att_mask = torch.cat(
            [encoder_attention_mask, text_embedding_mask], 1)

        answer_targets = (decoder_targets *
                          self.image_res).long() + self.bin_start_idx
        prefix_ids = torch.repeat_interleave(self.prefix_ids,
                                             merge_att_mask.shape[0],
                                             0).unsqueeze(-1)
        prefix_ids = torch.cat([prefix_ids, answer_targets], dim=1)

        answer_output = self.decoder(
            prefix_ids,
            encoder_hidden_states=merged_encode_hs,
            encoder_attention_mask=merge_att_mask,
            labels=None,
            return_dict=True,
        )
        prob_mask = self.bbox_prob_mask.view(1, 1,
                                             self.bbox_prob_mask.shape[-1])
        prediction_scores = answer_output.logits + prob_mask

        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = prefix_ids[:, 1:].contiguous()
        vocab_size = len(self.tokenizer)
        loss_seq_init = self.loss_fn(
            shifted_prediction_scores.view(-1, vocab_size), labels.view(-1))

        with torch.no_grad():
            pred_box = (torch.argmax(
                prediction_scores[:, :-1, :].contiguous(), dim=-1) -
                        self.bin_start_idx) / self.image_res
            weight_bbox = F.l1_loss(
                pred_box, decoder_targets, reduction='none').clamp(
                    0, 5) * self.box_l1_loss_coeff
            weight_giou = (1 - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(pred_box),
                    box_cxcywh_to_xyxy(decoder_targets)))
                           ) * self.box_giou_loss_coeff
            bs = text_embedding.shape[0]
            loss_seq = loss_seq_init[:].view(bs, -1, 4)
            loss_seq = loss_seq * weight_bbox
            loss_seq = loss_seq * weight_giou.unsqueeze(1)

        loss_seq = loss_seq.mean()

        losses = {
            'loss_seq': loss_seq,
            'loss_seq_init': loss_seq_init.mean(),
            'loss': loss_seq,
            'box_l1': weight_bbox.mean(-1).mean().detach(),
            'box_giou': weight_giou.mean().detach()
        }

        return losses

    def predict(
        self,
        text_embedding,
        text_embedding_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        """Generates the bbox coordinates at inference time."""

        merged_encode_hs = torch.cat([encoder_hidden_states, text_embedding],
                                     1)
        merge_att_mask = torch.cat(
            [encoder_attention_mask, text_embedding_mask], 1)

        prefix_ids = torch.repeat_interleave(self.prefix_ids,
                                             merge_att_mask.shape[0],
                                             0).unsqueeze(-1)

        for _ in range(4):
            decoder_output = self.decoder(
                prefix_ids,
                encoder_hidden_states=merged_encode_hs,
                encoder_attention_mask=merge_att_mask,
                labels=None,
                return_dict=True,
            )
            prob_mask = self.bbox_prob_mask.view(1, 1,
                                                 self.bbox_prob_mask.shape[-1])
            prediction_scores = decoder_output.logits + prob_mask

            prefix_ids = torch.cat([
                prefix_ids,
                torch.argmax(prediction_scores[:, -1, :], dim=-1).unsqueeze(1)
            ],
                                   dim=1)

        pred_box = self.process_bbox(prefix_ids[:, 1:])  # xywh 0-1 to xyxy 0-1

        return pred_box

    @torch.no_grad()
    def process_bbox(self, bbox):
        bbox = bbox - self.bin_start_idx
        bbox = torch.true_divide(bbox, self.image_res)
        bbox = box_cxcywh_to_xyxy(bbox)
        bbox = torch.clip(bbox, 0, 1)
        assert torch.all(bbox <= 1)
        return bbox
