# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmpretrain.registry import MODELS
from ..utils import (ChannelMultiheadAttention, PositionEncodingFourier,
                     build_norm_layer)
from .base_backbone import BaseBackbone
from .convnext import ConvNeXtBlock


class SDTAEncoder(BaseModule):
    """A PyTorch implementation of split depth-wise transpose attention (SDTA)
    encoder.

    Inspiration from
    https://github.com/mmaaz60/EdgeNeXt
    Args:
        in_channel (int): Number of input channels.
        drop_path_rate (float): Stochastic depth dropout rate.
            Defaults to 0.
        layer_scale_init_value (float): Initial value of layer scale.
            Defaults to 1e-6.
        mlp_ratio (int): Number of channels ratio in the MLP.
            Defaults to 4.
        use_pos_emb (bool): Whether to use position encoding.
            Defaults to True.
        num_heads (int): Number of heads in the multihead attention.
            Defaults to 8.
        qkv_bias (bool): Whether to use bias in the multihead attention.
            Defaults to True.
        attn_drop (float): Dropout rate of the attention.
            Defaults to 0.
        proj_drop (float): Dropout rate of the projection.
            Defaults to 0.
        layer_scale_init_value (float): Initial value of layer scale.
            Defaults to 1e-6.
        norm_cfg (dict): Dictionary to construct normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): Dictionary to construct activation layer.
            Defaults to ``dict(type='GELU')``.
        scales (int): Number of scales. Default to 1.
    """

    def __init__(self,
                 in_channel,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 mlp_ratio=4,
                 use_pos_emb=True,
                 num_heads=8,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 scales=1,
                 init_cfg=None):
        super(SDTAEncoder, self).__init__(init_cfg=init_cfg)
        conv_channels = max(
            int(math.ceil(in_channel / scales)),
            int(math.floor(in_channel // scales)))
        self.conv_channels = conv_channels
        self.num_convs = scales if scales == 1 else scales - 1

        self.conv_modules = ModuleList()
        for i in range(self.num_convs):
            self.conv_modules.append(
                nn.Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    padding=1,
                    groups=conv_channels))

        self.pos_embed = PositionEncodingFourier(
            embed_dims=in_channel) if use_pos_emb else None

        self.norm_csa = build_norm_layer(norm_cfg, in_channel)
        self.gamma_csa = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channel),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.csa = ChannelMultiheadAttention(
            embed_dims=in_channel,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop)

        self.norm = build_norm_layer(norm_cfg, in_channel)
        self.pointwise_conv1 = nn.Linear(in_channel, mlp_ratio * in_channel)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = nn.Linear(mlp_ratio * in_channel, in_channel)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channel),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        spx = torch.split(x, self.conv_channels, dim=1)
        for i in range(self.num_convs):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.conv_modules[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        x = torch.cat((out, spx[self.num_convs]), 1)

        # Channel Self-attention
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embed:
            pos_encoding = self.pos_embed((B, H, W))
            pos_encoding = pos_encoding.reshape(B, -1,
                                                x.shape[1]).permute(0, 2, 1)
            x += pos_encoding

        x = x + self.drop_path(self.gamma_csa * self.csa(self.norm_csa(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        x = shortcut + self.drop_path(x)

        return x


@MODELS.register_module()
class EdgeNeXt(BaseBackbone):
    """EdgeNeXt.

    A PyTorch implementation of: `EdgeNeXt: Efficiently Amalgamated
    CNN-Transformer Architecture for Mobile Vision Applications
    <https://arxiv.org/abs/2206.10589>`_

    Inspiration from
    https://github.com/mmaaz60/EdgeNeXt

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architectures in ``EdgeNeXt.arch_settings``.
            And if dict, it should include the following keys:

            - channels (list[int]): The number of channels at each stage.
            - depths (list[int]): The number of blocks at each stage.
            - num_heads (list[int]): The number of heads at each stage.

            Defaults to 'xxsmall'.
        in_channels (int): The number of input channels.
            Defaults to 3.
        global_blocks (list[int]): The number of global blocks.
            Defaults to [0, 1, 1, 1].
        global_block_type (list[str]): The type of global blocks.
            Defaults to ['None', 'SDTA', 'SDTA', 'SDTA'].
        drop_path_rate (float): Stochastic depth dropout rate.
            Defaults to 0.
        layer_scale_init_value (float): Initial value of layer scale.
            Defaults to 1e-6.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to False.
        mlp_ratio (int): The number of channel ratio in MLP layers.
            Defaults to 4.
        conv_kernel_size (list[int]): The kernel size of convolutional layers
            at each stage. Defaults to [3, 5, 7, 9].
        use_pos_embd_csa (list[bool]): Whether to use positional embedding in
            Channel Self-Attention. Defaults to [False, True, False, False].
        use_pos_emebd_global (bool): Whether to use positional embedding for
            whole network. Defaults to False.
        d2_scales (list[int]): The number of channel groups used for SDTA at
            each stage. Defaults to [2, 2, 3, 4].
        norm_cfg (dict): The config of normalization layer.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. Defaults to True.
        act_cfg (dict): The config of activation layer.
            Defaults to ``dict(type='GELU')``.
        init_cfg (dict, optional): Config for initialization.
            Defaults to None.
    """
    arch_settings = {
        'xxsmall': {  # parameters: 1.3M
            'channels': [24, 48, 88, 168],
            'depths': [2, 2, 6, 2],
            'num_heads': [4, 4, 4, 4]
        },
        'xsmall': {  # parameters: 2.3M
            'channels': [32, 64, 100, 192],
            'depths': [3, 3, 9, 3],
            'num_heads': [4, 4, 4, 4]
        },
        'small': {  # parameters: 5.6M
            'channels': [48, 96, 160, 304],
            'depths': [3, 3, 9, 3],
            'num_heads': [8, 8, 8, 8]
        },
        'base': {  # parameters: 18.51M
            'channels': [80, 160, 288, 584],
            'depths': [3, 3, 9, 3],
            'num_heads': [8, 8, 8, 8]
        },
    }

    def __init__(self,
                 arch='xxsmall',
                 in_channels=3,
                 global_blocks=[0, 1, 1, 1],
                 global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 linear_pw_conv=True,
                 mlp_ratio=4,
                 conv_kernel_sizes=[3, 5, 7, 9],
                 use_pos_embd_csa=[False, True, False, False],
                 use_pos_embd_global=False,
                 d2_scales=[2, 2, 3, 4],
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(EdgeNeXt, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_settings, \
                f'Arch {arch} is not in default archs ' \
                f'{set(self.arch_settings)}'
            self.arch_settings = self.arch_settings[arch]
        elif isinstance(arch, dict):
            essential_keys = {'channels', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.channels = self.arch_settings['channels']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.use_pos_embd_global = use_pos_embd_global

        for g in global_block_type:
            assert g in ['None',
                         'SDTA'], f'Global block type {g} is not supported'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        if self.use_pos_embd_global:
            self.pos_embed = PositionEncodingFourier(
                embed_dims=self.channels[0])
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]

        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=4, stride=4),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        self.stages = ModuleList()
        block_idx = 0
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2,
                    ))
                self.downsample_layers.append(downsample_layer)

            stage_blocks = []
            for j in range(depth):
                if j > depth - global_blocks[i] - 1:
                    stage_blocks.append(
                        SDTAEncoder(
                            in_channel=channels,
                            drop_path_rate=dpr[block_idx + j],
                            mlp_ratio=mlp_ratio,
                            scales=d2_scales[i],
                            use_pos_emb=use_pos_embd_csa[i],
                            num_heads=self.num_heads[i],
                        ))
                else:
                    dw_conv_cfg = dict(
                        kernel_size=conv_kernel_sizes[i],
                        padding=conv_kernel_sizes[i] // 2,
                    )
                    stage_blocks.append(
                        ConvNeXtBlock(
                            in_channels=channels,
                            dw_conv_cfg=dw_conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            linear_pw_conv=linear_pw_conv,
                            drop_path_rate=dpr[block_idx + j],
                            layer_scale_init_value=layer_scale_init_value,
                        ))
            block_idx += depth

            stage_blocks = Sequential(*stage_blocks)
            self.stages.append(stage_blocks)

            if i in self.out_indices:
                out_norm_cfg = dict(type='LN') if self.gap_before_final_norm \
                    else norm_cfg
                norm_layer = build_norm_layer(out_norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

    def init_weights(self) -> None:
        # TODO: need to be implemented in the future
        return super().init_weights()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if self.pos_embed and i == 0:
                B, _, H, W = x.shape
                x += self.pos_embed((B, H, W))

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap.flatten(1)))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(EdgeNeXt, self).train(mode)
        self._freeze_stages()
