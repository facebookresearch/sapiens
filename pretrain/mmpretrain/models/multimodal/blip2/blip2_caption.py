# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from mmengine.model import BaseModel
from torch import nn

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample


@MODELS.register_module()
class Blip2Caption(BaseModel):
    """BLIP2 Caption.

    Module for BLIP2 Caption task.

    Args:
        vision_backbone (dict): The config dict for vision backbone.
        text_backbone (dict): The config dict for text backbone.
        multimodal_backbone (dict): The config dict for multimodal backbone.
        vision_neck (dict): The config dict for vision neck.
        tokenizer: (Optional[dict]): The config for tokenizer.
            Defaults to None.
        prompt (str): Prompt used for training and eval.
            Defaults to ''.
        max_txt_len (int): Max text length of input text.
        num_captions (int): Number of captions to be generated for each image.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MultiModalDataPreprocessor" as type.
            See :class:`MultiModalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """
    _no_split_modules = ['BEiTViT', 'OPTDecoderLayer', 'BertLayer']

    def __init__(self,
                 vision_backbone: dict,
                 text_backbone: dict,
                 multimodal_backbone: dict,
                 vision_neck: dict,
                 tokenizer: Optional[dict] = None,
                 prompt: str = '',
                 max_txt_len: int = 20,
                 num_captions: int = 1,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.tokenizer = TOKENIZER.build(tokenizer)
        self.eos_token_id = self.tokenizer(
            '\n', add_special_tokens=False).input_ids[0]

        self.vision_backbone = MODELS.build(vision_backbone)
        self.ln_vision_backbone = nn.LayerNorm(self.vision_backbone.embed_dims)

        self.vision_neck = MODELS.build(vision_neck)

        self.text_backbone = MODELS.build(text_backbone)

        self.multimodal_backbone = MODELS.build(multimodal_backbone)
        self.multimodal_backbone.cls = None
        self.multimodal_backbone.bert.embeddings.word_embeddings = None
        self.multimodal_backbone.bert.embeddings.position_embeddings = None
        for layer in self.multimodal_backbone.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.prompt = prompt
        self.max_txt_len = max_txt_len
        self.num_captions = num_captions
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.multimodal_backbone.bert.config.query_length,
                        self.multimodal_backbone.bert.config.hidden_size))
        self.query_tokens.data.normal_(
            mean=0.0,
            std=self.multimodal_backbone.bert.config.initializer_range)

        # freeze the text backbone
        for _, param in self.text_backbone.named_parameters():
            param.requires_grad = False

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(
                self._ignore_loading_llm_keys_hook)

        if hasattr(self, '_register_state_dict_hook'):
            self._register_state_dict_hook(self._igonre_saving_llm_keys_hook)

    def forward(self,
                images: torch.Tensor,
                data_samples: Optional[List] = None,
                mode: str = 'loss'):
        """The unified entry for a forward process in both training and test.
        The method should accept two modes: "predict" and "loss":

        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            images (torch.Tensor): pre_processed img tensor  (N, C, ...).
            data_samples (List[DataSample], optional):
            mode (str): Return what kind of value. Defaults to 'loss'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
        """
        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def loss(self,
             images: torch.Tensor,
             data_samples: Optional[list] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            images (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``loss``
                method of :attr:`head`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        # extract image features
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

        self.tokenizer.padding_side = 'right'

        prompt = [
            self.prompt + data_sample.gt_caption + '\n'
            for data_sample in data_samples
        ]

        opt_tokens = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
        ).to(images.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.tokenizer.pad_token_id, -100)
        if self.prompt:
            targets[:, :self.prompt_length] = -100

        empty_targets = (
            torch.ones(attns_opt.size(),
                       dtype=torch.long).to(images.device).fill_(-100))
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = (
            self.text_backbone.model.decoder.embed_tokens(
                opt_tokens.input_ids))
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([attns_opt, opt_tokens.attention_mask],
                                   dim=1)

        outputs = self.text_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {'loss': loss}

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

        # extract image features
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

        prompt = [self.prompt] * image_embeds.size(0)

        opt_tokens = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
        ).to(images.device)
        attention_mask = torch.cat([attns_opt, opt_tokens.attention_mask],
                                   dim=1)

        inputs_embeds = (
            self.text_backbone.get_input_embeddings()(opt_tokens.input_ids))
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

        outputs = self.text_backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=0.9,
            temperature=1.,
            num_beams=5,
            max_new_tokens=self.max_txt_len,
            min_length=1,
            eos_token_id=self.eos_token_id,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_return_sequences=self.num_captions,
        )

        output_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(len(output_text))]

        for data_sample, decode_token in zip(data_samples, output_text):
            if data_sample is None:
                data_sample = DataSample()
            data_sample.pred_caption = decode_token
            out_data_samples.append(data_sample)

        return out_data_samples

    @staticmethod
    def _ignore_loading_llm_keys_hook(module, incompatible_keys):
        """Avoid warning missing keys of the LLM model."""
        import re
        llm_pattern = '^text_backbone'
        for key in list(incompatible_keys.missing_keys):
            if re.match(llm_pattern, key):
                incompatible_keys.missing_keys.remove(key)

    @staticmethod
    def _igonre_saving_llm_keys_hook(module, state_dict, prefix, metadata):
        """Avoid saving llm state dict."""
        import re
        llm_pattern = '^text_backbone'
        keys = [k for k, _ in state_dict.items()]
        for key in keys:
            if re.match(llm_pattern, key):
                state_dict.pop(key)
