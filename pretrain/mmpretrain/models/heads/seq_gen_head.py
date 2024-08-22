# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class SeqGenerationHead(BaseModule):
    """Generation head for multi-modal pre-trained task, adopted by BLIP.
    Normally used for generation task.

    Args:
        decoder (dict): Decoder for blip generation head.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        decoder: dict,
        ignore_index=-100,
        loss: dict = dict(type='LabelSmoothLoss', label_smooth_val=0.1),
        init_cfg: Optional[dict] = None,
    ) -> None:
        super(SeqGenerationHead, self).__init__(init_cfg=init_cfg)
        self.decoder = MODELS.build(decoder)
        self.loss_fn = MODELS.build(loss)
        self.ignore_index = ignore_index

    def forward(self, input_ids: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor, labels: torch.Tensor):
        """Forward to get decoder output.

        Args:
            input_ids (torch.Tensor): The tokenized input text tensor.
            encoder_hidden_states (torch.Tensor): Hidden states from image
                embeddings.
            encoder_attention_mask (torch.Tensor): Image embeddings hidden
                states attention mask.
            labels (torch.Tensor): Decoder target for calculate loss.

        Returns:
            dict[str, Tensor]: a dictionary of decoder outputs.
        """

        decoder_out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            return_dict=True,
        )
        return decoder_out

    def loss(self, input_ids, encoder_hidden_states, encoder_attention_mask,
             labels):
        """Calculate losses from the extracted features.

        Args:
            input_ids (torch.Tensor): The tokenized input text tensor.
            encoder_hidden_states (torch.Tensor): Hidden states from image
                embeddings.
            encoder_attention_mask (torch.Tensor): Image embeddings hidden
                states attention mask.
            labels (torch.Tensor): Decoder target for calculate loss.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """

        decoder_out = self(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
        )
        prediction_scores = decoder_out['logits']
        # we are doing next-token prediction;
        # shift prediction scores and input ids by one
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        vocab_size = prediction_scores.shape[-1]

        # mask ignored index
        if (labels == self.ignore_index).any():
            labels = labels.view(-1).clone()
            ignore_mask = (labels == self.ignore_index)
            labels.masked_fill_(ignore_mask, 0)
            weight = torch.logical_not(ignore_mask)
            avg_factor = max(weight.sum(), 1)
        else:
            weight = None
            avg_factor = labels.size(0)

        lm_loss = self.loss_fn(
            shifted_prediction_scores.view(-1, vocab_size),
            labels,
            weight=weight,
            avg_factor=avg_factor,
        )
        losses = {
            'seq_gen_lm_loss': lm_loss,
        }

        return losses

    def predict(self,
                input_ids,
                encoder_hidden_states,
                sep_token_id,
                pad_token_id,
                use_nucleus_sampling=False,
                num_beams=3,
                max_length=20,
                min_length=2,
                top_p=0.9,
                repetition_penalty=1.0,
                **kwargs):
        """Decoder prediction method.

        Args:
            input_ids (torch.Tensor): The tokenized input text tensor.
            encoder_hidden_states (torch.Tensor): Hidden states from image
                embeddings.
            sep_token_id (int): Tokenid of separation token.
            pad_token_id (int): Tokenid of pad token.
            use_nucleus_sampling (bool): Whether to use nucleus sampling in
                prediction. Defaults to False.
            num_beams (int): Number of beams used in predition.
                Defaults to 3.
            max_length (int): Max length of generated text in predition.
                Defaults to 20.
            min_length (int): Min length of generated text in predition.
                Defaults to 20.
            top_p (float):
                If < 1.0, only keep the top tokens with cumulative probability
                 >= top_p (nucleus filtering). Defaults to 0.9.
            repetition_penalty (float): The parameter for repetition penalty.
                Defaults to 1.0.
            **kwarg: Other arguments that might used in generation.

        Returns:
            dict[str, Tensor]: a dictionary of generation outputs.
        """
        device = encoder_hidden_states.device

        # TODO: In old version of transformers
        # Additional repeat interleave of hidden states should be add here.
        image_atts = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long).to(device)

        model_kwargs = {
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': image_atts,
        }
        model_kwargs.update(kwargs)

        if use_nucleus_sampling:
            # nucleus sampling
            outputs = self.decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=sep_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search
            outputs = self.decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=sep_token_id,
                pad_token_id=pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        return outputs
