# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class VQAGenerationHead(BaseModule):
    """Generation head for multi-modal pre-trained task, adapted by BLIP.
    Normally used for qa generation task (open-set)

    Args:
        decoder (dict): Decoder for decoding answers.
        inference_method (str): Inference method. One of 'rank', 'generate'.
            - If 'rank', the model will return answers with the highest
                probability from the answer list.
            - If 'generate', the model will generate answers.
            - Only for test, not for train / val.
        num_beams (int): Number of beams for beam search. 1 means no beam
            search. Only support when inference_method=='generate'.
            Defaults to 3.
        num_ans_candidates (int): Number of answer candidates, used to filter
            out answers with low probability. Only support when
            inference_method=='rank'. Defaults to 128.
        loss (dict or nn.Module): Config of loss or module of loss. Defaults to
            ``nn.CrossEntropyLoss(reduction='none', ignore_index=-100)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
        answer_list_path (str, optional): Path to `answer_list.json`
            (json file of a answer list). Required when
            inference_method=='rank'.


    TODO: `mmcls.LabelSmoothLoss` has not support `ignore_index` param.
    Now using `nn.CrossEntropyLoss`, without label_smoothing, in order to
    maintain compatibility with torch < 1.10.0
    """

    def __init__(
        self,
        decoder: dict,
        inference_method: str = 'generate',
        num_beams: int = 3,
        num_ans_candidates: int = 128,
        loss: Union[dict, nn.Module] = nn.CrossEntropyLoss(
            reduction='none', ignore_index=-100),
        init_cfg: Optional[dict] = None,
        answer_list_path: Optional[str] = None,
    ) -> None:

        super(VQAGenerationHead, self).__init__(init_cfg=init_cfg)
        self.decoder = MODELS.build(decoder)

        if inference_method == 'generate':
            assert isinstance(num_beams, int), \
                'for VQA `generate` mode, `num_beams` must be a int.'
            self.num_beams = num_beams
            self.num_ans_candidates = None
            self.answer_list = None

        elif inference_method == 'rank':
            assert isinstance(num_ans_candidates, int), \
                'for VQA `rank` mode, `num_ans_candidates` must be a int.'
            assert isinstance(answer_list_path, str), \
                'for VQA `rank` mode, `answer_list_path` must be set as ' \
                'the path to `answer_list.json`.'
            self.num_beams = None
            self.answer_list = mmengine.load(answer_list_path)
            if isinstance(self.answer_list, dict):
                self.answer_list = list(self.answer_list.keys())
            assert isinstance(self.answer_list, list) and all(
                isinstance(item, str) for item in self.answer_list), \
                'for VQA `rank` mode, `answer_list.json` must be a list of str'
            self.num_ans_candidates = min(num_ans_candidates,
                                          len(self.answer_list))

        else:
            raise AssertionError(
                'for VQA, `inference_method` must be "generate" or "rank", '
                'got {}.'.format(inference_method))

        self.inference_method = inference_method
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

    def forward(self, feats: dict):
        prediction_logits = self.decoder(
            feats['answer_input_ids'],
            attention_mask=feats['answer_attention_mask'],
            encoder_hidden_states=feats['question_states'],
            encoder_attention_mask=feats['question_atts'],
            labels=feats['answer_targets'],
            return_dict=True,
            return_logits=True,  # directly return logits, not computing loss
            reduction='none',
        )
        return prediction_logits

    def loss(self, feats: dict, data_samples=None):
        """Calculate losses from the extracted features.

        Args:
            feats (dict): The features extracted from the backbone.
            data_samples (List[BaseDataElement]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        shifted_prediction_scores = self(feats)
        labels = feats['answer_targets']
        lm_loss = None

        # we are doing next-token prediction;
        # shift prediction scores and input ids by one
        labels = labels[:, 1:].contiguous()
        lm_loss = self.loss_module(
            shifted_prediction_scores.view(-1,
                                           self.decoder.med_config.vocab_size),
            labels.view(-1))
        lm_loss = lm_loss.view(shifted_prediction_scores.size(0), -1).sum(1)
        # compute weighted loss
        losses = dict()
        loss = feats['answer_weight'] * lm_loss
        loss = loss.sum() / feats['batch_size']
        losses['vqa_loss'] = loss

        return losses

    def predict_rank(self, feats: dict, data_samples=None):
        """Predict rank in a close-set answer list."""
        question_states = feats['multimodal_embeds']
        question_atts = feats['question_atts']
        answer_candidates = feats['answer_candidates']
        assert answer_candidates is not None

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction='none',
        )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(
            logits, dim=1).index_select(
                dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(
            self.num_ans_candidates, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == feats['pad_token_id'],
                                            -100)

        def tile(x, dim, n_tile):
            init_dim = x.size(dim)
            repeat_idx = [1] * x.dim()
            repeat_idx[dim] = n_tile
            x = x.repeat(*(repeat_idx))
            order_index = torch.LongTensor(
                np.concatenate([
                    init_dim * np.arange(n_tile) + i for i in range(init_dim)
                ]))
            return torch.index_select(x, dim, order_index.to(x.device))

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, self.num_ans_candidates)
        question_atts = tile(question_atts, 0, self.num_ans_candidates)

        output = self.decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction='none',
        )

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, self.num_ans_candidates)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        answers = [self.answer_list[max_id] for max_id in max_ids]

        return answers

    def predict_generate(self, feats: dict, data_samples=None):
        """Predict answers in a generation manner."""
        device = feats['multimodal_embeds'].device
        question_states = feats['multimodal_embeds']
        question_atts = torch.ones(
            question_states.size()[:-1], dtype=torch.long).to(device)
        model_kwargs = {
            'encoder_hidden_states': question_states,
            'encoder_attention_mask': question_atts
        }

        bos_ids = torch.full((feats['multimodal_embeds'].shape[0], 1),
                             fill_value=feats['bos_token_id'],
                             device=device)

        outputs = self.decoder.generate(
            input_ids=bos_ids,
            max_length=10,
            min_length=1,
            num_beams=self.num_beams,
            eos_token_id=feats['sep_token_id'],
            pad_token_id=feats['pad_token_id'],
            **model_kwargs)

        return outputs

    def predict(self, feats: dict, data_samples=None):
        """Predict results from the extracted features."""
        if self.inference_method == 'generate':
            return self.predict_generate(feats, data_samples)
        elif self.inference_method == 'rank':
            return self.predict_rank(feats, data_samples)
