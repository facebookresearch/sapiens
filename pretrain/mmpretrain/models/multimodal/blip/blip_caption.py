# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample


@MODELS.register_module()
class BlipCaption(BaseModel):
    """BLIP Caption.

    Args:
        vision_encoder (dict): Encoder for extracting image features.
        decoder_head (dict): The decoder head module to forward and
            calculate loss from processed features.
        tokenizer: (Optional[dict]): The config for tokenizer.
            Defaults to None.
        prompt (str): Prompt used for training and eval.
            Defaults to ''.
        max_txt_len (int): Max text length of input text.
        num_captions (int): Number of captions to be generated for each image.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MutimodalDataPreprocessor" as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 vision_encoder: dict,
                 decoder_head: dict,
                 tokenizer: Optional[dict] = None,
                 prompt: str = '',
                 max_txt_len: int = 20,
                 num_captions: int = 1,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super(BlipCaption, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.tokenizer = TOKENIZER.build(tokenizer)
        self.visual_encoder = MODELS.build(vision_encoder)
        self.seq_gen_head = MODELS.build(decoder_head)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        self.max_txt_len = max_txt_len
        self.num_captions = num_captions

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List] = None,
        mode: str = 'loss',
    ):
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
            data_samples (List[DataSample], optional): Data samples with
                additional infos.
            mode (str): Return what kind of value. Defaults to 'loss'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def predict(self, images, data_samples=None, **kwargs):
        """Predict captions from a batch of inputs.

        Args:
            images (torch.Tensor): The input images tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        # prepare inputs for decoder generation.
        image_embeds = self.visual_encoder(images)[0]
        image_embeds = torch.repeat_interleave(image_embeds, self.num_captions,
                                               0)

        prompt = [self.prompt] * image_embeds.size(0)
        prompt = self.tokenizer(
            prompt, padding='longest',
            return_tensors='pt').to(image_embeds.device)

        prompt.input_ids[:, 0] = self.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        decoder_out = self.seq_gen_head.predict(
            input_ids=prompt.input_ids,
            encoder_hidden_states=image_embeds,
            sep_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        decode_tokens = self.tokenizer.batch_decode(
            decoder_out.sequences, skip_special_tokens=True)

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(len(decode_tokens))]

        for data_sample, decode_token in zip(data_samples, decode_tokens):
            if data_sample is None:
                data_sample = DataSample()
            data_sample.pred_caption = decode_token[len(self.prompt):]
            out_data_samples.append(data_sample)

        return out_data_samples

    def loss(self, images, data_samples):
        """Calculate losses from a batch of images and data samples.

        Args:
            images (torch.Tensor): The input images tensor with shape
                (N, C, ...) in general.
            data_samples (List[ImageTextDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        image_embeds = self.visual_encoder(images)[0]
        raw_text = [self.prompt + ds.gt_caption for ds in data_samples]

        text = self.tokenizer(
            raw_text,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors='pt',
        ).to(image_embeds.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # prepare targets for forwarding decoder
        labels = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100)
        labels[:, :self.prompt_length] = -100
        # forward decoder
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        losses = self.seq_gen_head.loss(
            input_ids=text.input_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=labels,
        )
        return losses
