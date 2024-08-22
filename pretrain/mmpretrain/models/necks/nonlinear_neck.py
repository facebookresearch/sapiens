# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class NonLinearNeck(BaseModule):
    """The non-linear neck.

    Structure: fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        num_layers: int = 2,
        with_bias: bool = False,
        with_last_bn: bool = True,
        with_last_bn_affine: bool = True,
        with_last_bias: bool = False,
        with_avg_pool: bool = True,
        norm_cfg: dict = dict(type='SyncBN'),
        init_cfg: Optional[Union[dict, List[dict]]] = [
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super(NonLinearNeck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = build_norm_layer(norm_cfg, hid_channels)[1]

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f'fc{i}')

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Forward function.

        Args:
            x (Tuple[torch.Tensor]): The feature map of backbone.

        Returns:
            Tuple[torch.Tensor]: The output features.
        """
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return (x, )
