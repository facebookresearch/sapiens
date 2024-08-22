# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.model import BaseModel

from mmpretrain.models.utils.box_utils import box_xyxy_to_cxcywh
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures.data_sample import DataSample


@MODELS.register_module()
class BlipGrounding(BaseModel):
    """BLIP Grounding.

    Args:
        visual_encoder (dict): Backbone for extracting image features.
        text_encoder (dict): Backbone for extracting text features.
                              but we integrate the vqa text extractor
                              into the tokenizer part in datasets/transform/
                              so we don't need text_backbone
        multimodal_encoder (Optional[dict]): Backbone for extracting
            multi-modal features. We apply this part as VQA fusion module.
        neck (Optional[dict]): The neck module to process features from
            backbone. Defaults to None.
        head (Optional[Union[List[dict], dict]]): The head module to calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MutimodalDataPreprocessor" as type.
            See :class:`MutimodalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 tokenizer: Optional[dict] = None,
                 visual_encoder: Optional[dict] = None,
                 text_encoder: Optional[dict] = None,
                 multimodal_encoder: Optional[dict] = None,
                 head: Optional[Union[List[dict], dict]] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super(BlipGrounding, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.tokenizer = TOKENIZER.build(tokenizer)
        self.prompt = 'localize instance: '
        self.visual_encoder = MODELS.build(visual_encoder)
        self.text_encoder = MODELS.build(text_encoder)
        self.multimodal_encoder = MODELS.build(multimodal_encoder)
        head.setdefault('tokenizer', self.tokenizer)
        self.grounding_head = MODELS.build(head)

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        mode: str = 'loss',
    ):
        """The unified entry for a forward process in both training and test.
        The method should accept only one mode "loss":

        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor, tuple): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[VQADataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
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

    def extract_feat(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
        Returns:
            image_embeds (Tensor): The output features.
        """
        image_embeds = self.visual_encoder(images)[0]
        return image_embeds

    def loss(
        self,
        images: torch.Tensor,
        data_samples=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """generate train_loss from the input tensor and data_samples.

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            data_samples (List[VQADataSample], optional): The annotation
                data of every samples..

        Returns:
            Dict[torch.Tensor]: The losses features.
        """

        # extract image feature
        image_embeds = self.extract_feat(images)
        image_atts = image_embeds.new_ones(
            image_embeds.size()[:-1], dtype=torch.long)

        raw_text = []
        box_targets = []
        for ds in data_samples:

            raw_text.append(ds.text)
            box_t = copy.deepcopy(ds.box) * 1.0
            box_t[1] /= ds.img_shape[0]
            box_t[3] /= ds.img_shape[0]
            box_t[0] /= ds.img_shape[1]
            box_t[2] /= ds.img_shape[1]

            box_targets.append(box_t)

        box_targets = image_embeds.new_tensor(np.stack(box_targets))
        box_targets = box_xyxy_to_cxcywh(box_targets)  # xywh 0-1

        text = self.tokenizer(
            raw_text,
            padding='longest',
            truncation=True,
            max_length=128,
            return_tensors='pt',
        ).to(image_embeds.device)

        text_embeds = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            mode='text',
            return_dict=True)  # bz, seq_len, hid

        # multimodal fusion
        multimodal_embeds = self.multimodal_encoder(
            encoder_embeds=text_embeds.last_hidden_state,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # put answer from data_samples into tensor form
        losses = self.grounding_head.loss(
            text_embedding=multimodal_embeds.last_hidden_state,
            text_embedding_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            decoder_targets=box_targets,
        )

        return losses

    def predict(self, images, data_samples=None):
        """"""

        # extract image feature
        image_embeds = self.extract_feat(images)
        image_atts = image_embeds.new_ones(
            image_embeds.size()[:-1], dtype=torch.long)

        raw_text = []
        for ds in data_samples:
            raw_text.append(ds.text)

        text = self.tokenizer(
            raw_text,
            padding='longest',
            truncation=True,
            max_length=128,
            return_tensors='pt',
        ).to(image_embeds.device)

        text_embeds = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            mode='text',
            return_dict=True)  # bz, seq_len, hid

        # multimodal fusion
        multimodal_embeds = self.multimodal_encoder(
            encoder_embeds=text_embeds.last_hidden_state,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # put answer from data_samples into tensor form
        output_boxes = self.grounding_head.predict(
            text_embedding=multimodal_embeds.last_hidden_state,
            text_embedding_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
        )  # xyxy 0-1

        out_data_samples = []
        for bbox, data_sample, img in zip(output_boxes, data_samples, images):
            if data_sample is None:
                data_sample = DataSample()

            img_size = img.shape[-2:]
            scale_factor = data_sample.get('scale_factor', (1, 1))
            bbox[0::2] = bbox[0::2] * img_size[1] / scale_factor[0]
            bbox[1::2] = bbox[1::2] * img_size[0] / scale_factor[1]
            bbox = bbox[None, :]
            data_sample.pred_bboxes = bbox

            if 'gt_bboxes' in data_sample:
                gt_bboxes = torch.Tensor(data_sample.get('gt_bboxes'))
                gt_bboxes[:, 0::2] /= scale_factor[0]
                gt_bboxes[:, 1::2] /= scale_factor[1]
                data_sample.gt_bboxes = gt_bboxes

            out_data_samples.append(data_sample)

        return out_data_samples
