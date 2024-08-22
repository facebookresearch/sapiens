# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER


@MODELS.register_module()
class BlipNLVR(BaseModel):
    """BLIP NLVR.

    Args:
        vision_backbone (dict): Backbone for extracting image features.
        text_backbone (dict): Backbone for extracting text features.
            but we integrate the vqa text extractor into the tokenizer part in
            datasets/transform/ so we don't need text_backbone
        multimodal_backbone (Optional[dict]): Backbone for extracting
            multi-modal features. We apply this part as VQA fusion module.
        neck (Optional[dict]): The neck module to process features from
            backbone. Defaults to None.
        head (Optional[dict]): The head module to calculate
            loss from processed features. See :mod:`mmmultimodal.models.heads`.
            Notice that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        tokenizer: (Optional[dict]): The config for tokenizer
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MutimodalDataPreprocessor" as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 vision_backbone: dict,
                 multimodal_backbone: dict,
                 tokenizer: Optional[dict] = None,
                 max_txt_len: int = 35,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        if tokenizer is not None:
            self.tokenizer = TOKENIZER.build(tokenizer)
        self.vision_backbone = MODELS.build(vision_backbone)
        self.multimodal_backbone = MODELS.build(multimodal_backbone)
        self.max_txt_len = max_txt_len

        # For simplity, directly use head definition here.
        # If more complex head is designed, move this and loss to a new
        # head module.
        hidden_size = self.multimodal_backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def preprocess_text(self, data_samples):

        sample_item = data_samples[0]

        if sample_item is not None and 'text' in sample_item:
            texts = [sample.get('text') for sample in data_samples]
        else:
            return None

        # perform tokenize first if satisfied conditions
        texts = self.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors='pt',
        ).to(self.device)

        return texts

    def forward(
        self,
        images: dict,
        data_samples: Optional[List] = None,
        mode: str = 'tensor',
    ):
        """The unified entry for a forward process in both training and test.
        The method should accept only one mode "loss":

        - "loss": Forward and return a dict of losses according to the given
          images and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            images (dict of torch.Tensor):
                img: pre_processed img tensor  (N, C, ...).
                text: tokenized text (N, L)
            data_samples (List[CaptionDataSample], optional):
            The annotation data of every samples.
                'image': raw image data
                'text' tokenized text
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        # B, T, C, H, W to T*B, C, H, W
        images = images.permute(1, 0, 2, 3, 4).flatten(0, 1)

        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def predict(self, images, data_samples=None):
        """Predict caption."""
        # prepare inputs for decoder generation.
        image_embeds = self.vision_backbone(images)[0]
        texts = self.preprocess_text(data_samples)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)

        image0_embeds, image1_embeds = torch.split(image_embeds,
                                                   texts.input_ids.size(0))

        # multimodal fusion
        multimodal_embeds = self.multimodal_backbone(
            texts.input_ids,
            attention_mask=texts.attention_mask,
            encoder_hidden_states=[image0_embeds, image1_embeds],
            encoder_attention_mask=[
                image_atts[:image0_embeds.size(0)],
                image_atts[image0_embeds.size(0):],
            ],
            return_dict=True,
        )

        # get prediction
        outputs = self.head(multimodal_embeds.last_hidden_state[:, 0, :])

        pred_scores = F.softmax(outputs, dim=1)

        for pred_score, data_sample in zip(pred_scores, data_samples):
            data_sample.set_pred_score(pred_score)
            data_sample.set_pred_label(pred_score.argmax(dim=0))

        return data_samples

    def loss(self, images, data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            images (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ImageTextDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # prepare inputs for decoder generation.
        image_embeds = self.vision_backbone(images)[0]
        texts = self.preprocess_text(data_samples)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        image0_embeds, image1_embeds = torch.split(image_embeds,
                                                   texts.input_ids.size(0))

        # multimodal fusion
        multimodal_embeds = self.multimodal_backbone(
            texts.input_ids,
            attention_mask=texts.attention_mask,
            encoder_hidden_states=[image0_embeds, image1_embeds],
            encoder_attention_mask=[
                image_atts[:image0_embeds.size(0)],
                image_atts[image0_embeds.size(0):],
            ],
            return_dict=True,
        )

        # get prediction
        outputs = self.head(multimodal_embeds.last_hidden_state[:, 0, :])

        targets = torch.tensor([i.gt_label
                                for i in data_samples]).to(outputs.device)
        loss = F.cross_entropy(outputs, targets)
        return {'loss': loss}
