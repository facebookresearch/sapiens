# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmengine.model import Sequential
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from .cls_head import ClsHead


@MODELS.register_module()
class VisionTransformerClsHead(ClsHead):
    """Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int, optional): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to ``dict(type='Tanh')``.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_dim: Optional[int] = None,
                 act_cfg: dict = dict(type='Tanh'),
                 init_cfg: dict = dict(type='Constant', layer='Linear', val=0),
                 **kwargs):
        super(VisionTransformerClsHead, self).__init__(
            init_cfg=init_cfg, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        """"Init hidden layer if exists."""
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
        self.layers = Sequential(OrderedDict(layers))

    def init_weights(self):
        """"Init weights of hidden layer if exists."""
        super(VisionTransformerClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

    def pre_logits(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage. In ``VisionTransformerClsHead``, we
        obtain the feature of the last stage and forward in hidden layer if
        exists.
        """
        feat = feats[-1]  # Obtain feature of the last scale.
        # For backward-compatibility with the previous ViT output
        cls_token = feat[-1] if isinstance(feat, list) else feat
        if self.hidden_dim is None:
            return cls_token
        else:
            x = self.layers.pre_logits(cls_token)
            return self.layers.act(x)

    def forward(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.layers.head(pre_logits)
        return cls_score
