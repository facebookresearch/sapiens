# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class LinearNeck(BaseModule):
    """Linear neck with Dimension projection.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        gap_dim (int): Dimensions of each sample channel, can be one of
            {0, 1, 2, 3}. Defaults to 0.
        norm_cfg (dict, optional): dictionary to construct and
            config norm layer. Defaults to dict(type='BN1d').
        act_cfg (dict, optional): dictionary to construct and
            config activate layer. Defaults to None.
        init_cfg (dict, optional): dictionary to initialize weights.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 gap_dim: int = 0,
                 norm_cfg: Optional[dict] = dict(type='BN1d'),
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.act_cfg = copy.deepcopy(act_cfg)

        assert gap_dim in [0, 1, 2, 3], 'GlobalAveragePooling dim only ' \
            f'support {0, 1, 2, 3}, get {gap_dim} instead.'
        if gap_dim == 0:
            self.gap = nn.Identity()
        elif gap_dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif gap_dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_dim == 3:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        else:
            self.norm = nn.Identity()

        if act_cfg:
            self.act = build_activation_layer(act_cfg)
        else:
            self.act = nn.Identity()

    def forward(self, inputs: Union[Tuple,
                                    torch.Tensor]) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Union[Tuple, torch.Tensor]): The features extracted from
                the backbone. Multiple stage inputs are acceptable but only
                the last stage will be used.

        Returns:
            Tuple[torch.Tensor]: A tuple of output features.
        """
        assert isinstance(inputs, (tuple, torch.Tensor)), (
            'The inputs of `LinearNeck` must be tuple or `torch.Tensor`, '
            f'but get {type(inputs)}.')
        if isinstance(inputs, tuple):
            inputs = inputs[-1]

        x = self.gap(inputs)
        x = x.view(x.size(0), -1)
        out = self.act(self.norm(self.fc(x)))
        return (out, )
