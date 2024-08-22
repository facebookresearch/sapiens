# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .cls_head import ClsHead


@MODELS.register_module()
class EfficientFormerClsHead(ClsHead):
    """EfficientFormer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        distillation (bool): Whether use a additional distilled head.
            Defaults to True.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 distillation=True,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(EfficientFormerClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dist = distillation

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.head = nn.Linear(self.in_channels, self.num_classes)
        if self.dist:
            self.dist_head = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.head(pre_logits)

        if self.dist:
            cls_score = (cls_score + self.dist_head(pre_logits)) / 2
        return cls_score

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In :obj`EfficientFormerClsHead`, we just
        obtain the feature of the last stage.
        """
        # The EfficientFormerClsHead doesn't have other module, just return
        # after unpacking.
        return feats[-1]

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.dist:
            raise NotImplementedError(
                "MMPretrain doesn't support to train"
                ' the distilled version EfficientFormer.')
        else:
            return super().loss(feats, data_samples, **kwargs)
