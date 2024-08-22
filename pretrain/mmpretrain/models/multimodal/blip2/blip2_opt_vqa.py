# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .blip2_caption import Blip2Caption


@MODELS.register_module()
class Blip2VQA(Blip2Caption):
    """BLIP2 VQA.

    Module for BLIP2 VQA task. For more details about the initialization
    params, please refer to :class:`Blip2Caption`.
    """

    def predict(self,
                images: torch.Tensor,
                data_samples: Optional[list] = None,
                **kwargs) -> List[DataSample]:
        """Predict captions from a batch of inputs.

        Args:
            images (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        questions = [d.question for d in data_samples]

        # extract image features from
        image_embeds = self.ln_vision_backbone(self.vision_backbone(images)[0])
        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
        ).to(images.device)

        # distill image features to query tokens
        query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1)
        query_outputs = self.multimodal_backbone.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_opt = self.vision_neck([query_outputs.last_hidden_state])
        attns_opt = torch.ones(
            inputs_opt.size()[:-1], dtype=torch.long).to(images.device)

        prompt = [self.prompt.format(q) for q in questions]

        # use left padding
        self.tokenizer.padding_side = 'left'

        opt_tokens = self.tokenizer(
            prompt, return_tensors='pt', padding='longest').to(images.device)
        input_ids = opt_tokens.input_ids
        attention_mask = torch.cat([attns_opt, opt_tokens.attention_mask],
                                   dim=1)

        inputs_embeds = self.text_backbone.model.decoder.embed_tokens(
            input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

        outputs = self.text_backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=5,
            max_new_tokens=self.max_txt_len,
            min_length=1,
            eos_token_id=self.eos_token_id,
            length_penalty=-1.0,
        )

        output_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        out_data_samples = []
        for data_sample, decode_token in zip(data_samples, output_text):
            data_sample.pred_answer = decode_token
            out_data_samples.append(data_sample)

        return out_data_samples
