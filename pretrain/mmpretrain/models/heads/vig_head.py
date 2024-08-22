# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer

from mmpretrain.registry import MODELS
from .cls_head import ClsHead


@MODELS.register_module()
class VigClsHead(ClsHead):
    """The classification head for Vision GNN.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int): The number of middle channels. Defaults to 1024.
        act_cfg (dict): The config of activation function.
            Defaults to ``dict(type='GELU')``.
        dropout (float): The dropout rate.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_dim: int = 1024,
                 act_cfg: dict = dict(type='GELU'),
                 dropout: float = 0.,
                 **kwargs):
        super().__init__(**kwargs)

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a stage_blocks stage. In ``VigClsHead``, we just obtain the
        feature of the last stage.
        """
        feats = feats[-1]
        feats = self.fc1(feats)
        feats = self.bn(feats)
        feats = self.act(feats)
        feats = self.drop(feats)

        return feats

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc2(pre_logits)
        return cls_score
