# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from ..backbones.resnet import Bottleneck, ResLayer


@MODELS.register_module()
class HRFuseScales(BaseModule):
    """Fuse feature map of multiple scales in HRNet.

    Args:
        in_channels (list[int]): The input channels of all scales.
        out_channels (int): The channels of fused feature map.
            Defaults to 2048.
        norm_cfg (dict): dictionary to construct norm layers.
            Defaults to ``dict(type='BN', momentum=0.1)``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01))``.
    """

    def __init__(self,
                 in_channels,
                 out_channels=2048,
                 norm_cfg=dict(type='BN', momentum=0.1),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(HRFuseScales, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg

        block_type = Bottleneck
        out_channels = [128, 256, 512, 1024]

        # Increase the channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        increase_layers = []
        for i in range(len(in_channels)):
            increase_layers.append(
                ResLayer(
                    block_type,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    num_blocks=1,
                    stride=1,
                ))
        self.increase_layers = nn.ModuleList(increase_layers)

        # Downsample feature maps in each scale.
        downsample_layers = []
        for i in range(len(in_channels) - 1):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                ))
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # The final conv block before final classifier linear layer.
        self.final_layer = ConvModule(
            in_channels=out_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )

    def forward(self, x):
        assert isinstance(x, tuple) and len(x) == len(self.in_channels)

        feat = self.increase_layers[0](x[0])
        for i in range(len(self.downsample_layers)):
            feat = self.downsample_layers[i](feat) + \
                self.increase_layers[i + 1](x[i + 1])

        return (self.final_layer(feat), )
