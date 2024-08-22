# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from official impl at https://github.com/raoyongming/HorNet.
try:
    import torch.fft
    fft = True
except ImportError:
    fft = None

import copy
from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks import DropPath

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from ..utils import LayerScale


def get_dwconv(dim, kernel_size, bias=True):
    """build a pepth-wise convolution."""
    return nn.Conv2d(
        dim,
        dim,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2,
        bias=bias,
        groups=dim)


class HorNetLayerNorm(nn.Module):
    """An implementation of LayerNorm of HorNet.

    The differences between HorNetLayerNorm & torch LayerNorm:
        1. Supports two data formats channels_last or channels_first.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an
            expected input of size.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        data_format (str): The ordering of the dimensions in the inputs.
            channels_last corresponds to inputs with shape (batch_size, height,
            width, channels) while channels_first corresponds to inputs with
            shape (batch_size, channels, height, width).
            Defaults to 'channels_last'.
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise ValueError(
                'data_format must be channels_last or channels_first')
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GlobalLocalFilter(nn.Module):
    """A GlobalLocalFilter of HorNet.

    Args:
        dim (int): Number of input channels.
        h (int): Height of complex_weight.
            Defaults to 14.
        w (int): Width of complex_weight.
            Defaults to 8.
    """

    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(
            dim // 2,
            dim // 2,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=dim // 2)
        self.complex_weight = nn.Parameter(
            torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        self.pre_norm = HorNetLayerNorm(
            dim, eps=1e-6, data_format='channels_first')
        self.post_norm = HorNetLayerNorm(
            dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(
                weight.permute(3, 0, 1, 2),
                size=x2.shape[2:4],
                mode='bilinear',
                align_corners=True).permute(1, 2, 3, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)],
                      dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnConv(nn.Module):
    """A gnConv of HorNet.

    Args:
        dim (int): Number of input channels.
        order (int): Order of gnConv.
            Defaults to 5.
        dw_cfg (dict): The Config for dw conv.
            Defaults to ``dict(type='DW', kernel_size=7)``.
        scale (float): Scaling parameter of gflayer outputs.
            Defaults to 1.0.
    """

    def __init__(self,
                 dim,
                 order=5,
                 dw_cfg=dict(type='DW', kernel_size=7),
                 scale=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        cfg = copy.deepcopy(dw_cfg)
        dw_type = cfg.pop('type')
        assert dw_type in ['DW', 'GF'],\
            'dw_type should be `DW` or `GF`'
        if dw_type == 'DW':
            self.dwconv = get_dwconv(sum(self.dims), **cfg)
        elif dw_type == 'GF':
            self.dwconv = GlobalLocalFilter(sum(self.dims), **cfg)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.projs = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], 1)
            for i in range(order - 1)
        ])

        self.scale = scale

    def forward(self, x):
        x = self.proj_in(x)
        y, x = torch.split(x, (self.dims[0], sum(self.dims)), dim=1)

        x = self.dwconv(x) * self.scale

        dw_list = torch.split(x, self.dims, dim=1)
        x = y * dw_list[0]

        for i in range(self.order - 1):
            x = self.projs[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class HorNetBlock(nn.Module):
    """A block of HorNet.

    Args:
        dim (int): Number of input channels.
        order (int): Order of gnConv.
            Defaults to 5.
        dw_cfg (dict): The Config for dw conv.
            Defaults to ``dict(type='DW', kernel_size=7)``.
        scale (float): Scaling parameter of gflayer outputs.
            Defaults to 1.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        use_layer_scale (bool): Whether to use use_layer_scale in HorNet
             block. Defaults to True.
    """

    def __init__(self,
                 dim,
                 order=5,
                 dw_cfg=dict(type='DW', kernel_size=7),
                 scale=1.0,
                 drop_path_rate=0.,
                 use_layer_scale=True):
        super().__init__()
        self.out_channels = dim

        self.norm1 = HorNetLayerNorm(
            dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnConv(dim, order, dw_cfg, scale)
        self.norm2 = HorNetLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if use_layer_scale:
            self.gamma1 = LayerScale(dim, data_format='channels_first')
            self.gamma2 = LayerScale(dim)
        else:
            self.gamma1, self.gamma2 = nn.Identity(), nn.Identity()

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.gamma1(self.gnconv(self.norm1(x))))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


@MODELS.register_module()
class HorNet(BaseBackbone):
    """HorNet backbone.

    A PyTorch implementation of paper `HorNet: Efficient High-Order Spatial
    Interactions with Recursive Gated Convolutions
    <https://arxiv.org/abs/2207.14284>`_ .
    Inspiration from https://github.com/raoyongming/HorNet

    Args:
        arch (str | dict): HorNet architecture.

            If use string, choose from 'tiny', 'small', 'base' and 'large'.
            If use dict, it should have below keys:

            - **base_dim** (int): The base dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **orders** (List[int]): The number of order of gnConv in each
                stage.
            - **dw_cfg** (List[dict]): The Config for dw conv.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        scale (float): Scaling parameter of gflayer outputs. Defaults to 1/3.
        use_layer_scale (bool): Whether to use use_layer_scale in HorNet
            block. Defaults to True.
        out_indices (Sequence[int]): Output from which stages.
            Default: ``(3, )``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'base_dim': 64,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [dict(type='DW', kernel_size=7)] * 4}),
        **dict.fromkeys(['t-gf', 'tiny-gf'],
                        {'base_dim': 64,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [
                             dict(type='DW', kernel_size=7),
                             dict(type='DW', kernel_size=7),
                             dict(type='GF', h=14, w=8),
                             dict(type='GF', h=7, w=4)]}),
        **dict.fromkeys(['s', 'small'],
                        {'base_dim': 96,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [dict(type='DW', kernel_size=7)] * 4}),
        **dict.fromkeys(['s-gf', 'small-gf'],
                        {'base_dim': 96,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [
                             dict(type='DW', kernel_size=7),
                             dict(type='DW', kernel_size=7),
                             dict(type='GF', h=14, w=8),
                             dict(type='GF', h=7, w=4)]}),
        **dict.fromkeys(['b', 'base'],
                        {'base_dim': 128,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [dict(type='DW', kernel_size=7)] * 4}),
        **dict.fromkeys(['b-gf', 'base-gf'],
                        {'base_dim': 128,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [
                             dict(type='DW', kernel_size=7),
                             dict(type='DW', kernel_size=7),
                             dict(type='GF', h=14, w=8),
                             dict(type='GF', h=7, w=4)]}),
        **dict.fromkeys(['b-gf384', 'base-gf384'],
                        {'base_dim': 128,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [
                             dict(type='DW', kernel_size=7),
                             dict(type='DW', kernel_size=7),
                             dict(type='GF', h=24, w=12),
                             dict(type='GF', h=13, w=7)]}),
        **dict.fromkeys(['l', 'large'],
                        {'base_dim': 192,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [dict(type='DW', kernel_size=7)] * 4}),
        **dict.fromkeys(['l-gf', 'large-gf'],
                        {'base_dim': 192,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [
                             dict(type='DW', kernel_size=7),
                             dict(type='DW', kernel_size=7),
                             dict(type='GF', h=14, w=8),
                             dict(type='GF', h=7, w=4)]}),
        **dict.fromkeys(['l-gf384', 'large-gf384'],
                        {'base_dim': 192,
                         'depths': [2, 3, 18, 2],
                         'orders': [2, 3, 4, 5],
                         'dw_cfg': [
                             dict(type='DW', kernel_size=7),
                             dict(type='DW', kernel_size=7),
                             dict(type='GF', h=24, w=12),
                             dict(type='GF', h=13, w=7)]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 drop_path_rate=0.,
                 scale=1 / 3,
                 use_layer_scale=True,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 with_cp=False,
                 gap_before_final_norm=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if fft is None:
            raise RuntimeError(
                'Failed to import torch.fft. Please install "torch>=1.7".')

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'base_dim', 'depths', 'orders', 'dw_cfg'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.scale = scale
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.gap_before_final_norm = gap_before_final_norm

        base_dim = self.arch_settings['base_dim']
        dims = list(map(lambda x: 2**x * base_dim, range(4)))

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            HorNetLayerNorm(dims[0], eps=1e-6, data_format='channels_first'))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                HorNetLayerNorm(
                    dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        total_depth = sum(self.arch_settings['depths'])
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                HorNetBlock(
                    dim=dims[i],
                    order=self.arch_settings['orders'][i],
                    dw_cfg=self.arch_settings['dw_cfg'][i],
                    scale=self.scale,
                    drop_path_rate=dpr[cur_block_idx + j],
                    use_layer_scale=use_layer_scale)
                for j in range(self.arch_settings['depths'][i])
            ])
            self.stages.append(stage)
            cur_block_idx += self.arch_settings['depths'][i]

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.stages) + index
            assert 0 <= out_indices[i] <= len(self.stages), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

        norm_layer = partial(
            HorNetLayerNorm, eps=1e-6, data_format='channels_first')
        for i_layer in out_indices:
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        super(HorNet, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = self.downsample_layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze blocks
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i in self.out_indices:
                # freeze norm
                m = getattr(self, f'norm{i + 1}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.with_cp:
                x = checkpoint.checkpoint_sequential(self.stages[i],
                                                     len(self.stages[i]), x)
            else:
                x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        return tuple(outs)
