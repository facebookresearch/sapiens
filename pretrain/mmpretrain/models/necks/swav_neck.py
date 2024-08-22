# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class SwAVNeck(BaseModule):
    """The non-linear neck of SwAV: fc-bn-relu-fc-normalization.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        with_avg_pool: bool = True,
        with_l2norm: bool = True,
        norm_cfg: dict = dict(type='SyncBN'),
        init_cfg: Optional[Union[dict, List[dict]]] = [
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super().__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.with_l2norm = with_l2norm
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if out_channels == 0:
            self.projection_neck = nn.Identity()
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            self.norm = build_norm_layer(norm_cfg, hid_channels)[1]
            self.projection_neck = nn.Sequential(
                nn.Linear(in_channels, hid_channels),
                self.norm,
                nn.ReLU(inplace=True),
                nn.Linear(hid_channels, out_channels),
            )

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Compute projection.

        Args:
            x (torch.Tensor): The feature vectors after pooling.

        Returns:
            torch.Tensor: The output features with projection or L2-norm.
        """
        x = self.projection_neck(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward function.

        Args:
            x (List[torch.Tensor]): list of feature maps, len(x) according to
                len(num_crops).

        Returns:
            torch.Tensor: The projection vectors.
        """
        avg_out = []
        for _x in x:
            _x = _x[0]
            if self.with_avg_pool:
                _out = self.avgpool(_x)
                avg_out.append(_out)
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output = self.forward_projection(feat_vec)
        return output
