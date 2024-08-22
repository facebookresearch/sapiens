# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from .cls_head import ClsHead


class LinearBlock(BaseModule):
    """Linear block for StackedLinearClsHead."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fc = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """The forward process."""
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


@MODELS.register_module()
class StackedLinearClsHead(ClsHead):
    """Classifier head with several hidden fc layer and a output fc layer.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence[int]): Number of channels in the hidden fc
            layers.
        dropout_rate (float): Dropout rate after each hidden fc layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each hidden fc layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mid_channels: Sequence[int],
                 dropout_rate: float = 0.,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 **kwargs):
        super(StackedLinearClsHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.in_channels = in_channels
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        assert isinstance(mid_channels, Sequence), \
            f'`mid_channels` of StackedLinearClsHead should be a sequence, ' \
            f'instead of {type(mid_channels)}'
        self.mid_channels = mid_channels

        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers()

    def _init_layers(self):
        """"Init layers."""
        self.layers = ModuleList()
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                LinearBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            in_channels = hidden_channels

        self.layers.append(
            LinearBlock(
                self.mid_channels[-1],
                self.num_classes,
                dropout_rate=0.,
                norm_cfg=None,
                act_cfg=None))

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage.
        """
        x = feats[-1]
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    @property
    def fc(self):
        """Full connected layer."""
        return self.layers[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
