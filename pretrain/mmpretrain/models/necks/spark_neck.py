# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from ..utils import build_norm_layer


def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


class ConvBlock2x(BaseModule):
    """The definition of convolution block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 norm_cfg: dict,
                 act_cfg: dict,
                 last_act: bool,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False)
        self.norm1 = build_norm_layer(norm_cfg, mid_channels)
        self.activate1 = MODELS.build(act_cfg)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = build_norm_layer(norm_cfg, out_channels)
        self.activate2 = MODELS.build(act_cfg) if last_act else nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activate1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activate2(out)
        return out


class DecoderConvModule(BaseModule):
    """The convolution module of decoder with upsampling."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 kernel_size: int = 4,
                 scale_factor: int = 2,
                 num_conv_blocks: int = 1,
                 norm_cfg: dict = dict(type='SyncBN'),
                 act_cfg: dict = dict(type='ReLU6'),
                 last_act: bool = True,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        padding = (kernel_size - scale_factor) // 2
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=padding,
            bias=True)

        conv_blocks_list = [
            ConvBlock2x(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                norm_cfg=norm_cfg,
                last_act=last_act,
                act_cfg=act_cfg) for _ in range(num_conv_blocks)
        ]
        self.conv_blocks = nn.Sequential(*conv_blocks_list)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv_blocks(x)


@MODELS.register_module()
class SparKLightDecoder(BaseModule):
    """The decoder for SparK, which upsamples the feature maps.

    Args:
        feature_dim (int): The dimension of feature map.
        upsample_ratio (int): The ratio of upsample, equal to downsample_raito
            of the algorithm.
        mid_channels (int): The middle channel of `DecoderConvModule`. Defaults
            to 0.
        kernel_size (int): The kernel size of `ConvTranspose2d` in
            `DecoderConvModule`. Defaults to 4.
        scale_factor (int): The scale_factor of `ConvTranspose2d` in
            `DecoderConvModule`. Defaults to 2.
        num_conv_blocks (int): The number of convolution blocks in
            `DecoderConvModule`. Defaults to 1.
        norm_cfg (dict): Normalization config. Defaults to dict(type='SyncBN').
        act_cfg (dict): Activation config. Defaults to dict(type='ReLU6').
        last_act (bool): Whether apply the last activation in
            `DecoderConvModule`. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        feature_dim: int,
        upsample_ratio: int,
        mid_channels: int = 0,
        kernel_size: int = 4,
        scale_factor: int = 2,
        num_conv_blocks: int = 1,
        norm_cfg: dict = dict(type='SyncBN'),
        act_cfg: dict = dict(type='ReLU6'),
        last_act: bool = False,
        init_cfg: Optional[dict] = [
            dict(type='Kaiming', layer=['Conv2d', 'ConvTranspose2d']),
            dict(type='TruncNormal', std=0.02, layer=['Linear']),
            dict(
                type='Constant',
                val=1,
                layer=['_BatchNorm', 'LayerNorm', 'SyncBatchNorm'])
        ],
    ):
        super().__init__(init_cfg=init_cfg)
        self.feature_dim = feature_dim

        assert is_pow2n(upsample_ratio)
        n = round(math.log2(upsample_ratio))
        channels = [feature_dim // 2**i for i in range(n + 1)]

        self.decoder = nn.ModuleList([
            DecoderConvModule(
                in_channels=c_in,
                out_channels=c_out,
                mid_channels=c_in if mid_channels == 0 else mid_channels,
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                num_conv_blocks=num_conv_blocks,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                last_act=last_act)
            for (c_in, c_out) in zip(channels[:-1], channels[1:])
        ])
        self.proj = nn.Conv2d(
            channels[-1], 3, kernel_size=1, stride=1, bias=True)

    def forward(self, to_dec):
        x = 0
        for i, d in enumerate(self.decoder):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.decoder[i](x)
        return self.proj(x)
