# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential
from torch.nn import functional as F

from mmpretrain.registry import MODELS
from ..utils import LeAttention
from .base_backbone import BaseBackbone


class ConvBN2d(Sequential):
    """An implementation of Conv2d + BatchNorm2d with support of fusion.

    Modified from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolution kernel.
            Default: 1.
        stride (int): The stride of the convolution.
            Default: 1.
        padding (int): The padding of the convolution.
            Default: 0.
        dilation (int): The dilation of the convolution.
            Default: 1.
        groups (int): The number of groups in the convolution.
            Default: 1.
        bn_weight_init (float): The initial value of the weight of
            the nn.BatchNorm2d layer. Default: 1.0.
        init_cfg (dict): The initialization config of the module.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.add_module(
            'conv2d',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False))
        bn2d = nn.BatchNorm2d(num_features=out_channels)
        # bn initialization
        torch.nn.init.constant_(bn2d.weight, bn_weight_init)
        torch.nn.init.constant_(bn2d.bias, 0)
        self.add_module('bn2d', bn2d)

    @torch.no_grad()
    def fuse(self):
        conv2d, bn2d = self._modules.values()
        w = bn2d.weight / (bn2d.running_var + bn2d.eps)**0.5
        w = conv2d.weight * w[:, None, None, None]
        b = bn2d.bias - bn2d.running_mean * bn2d.weight / \
            (bn2d.running_var + bn2d.eps)**0.5

        m = nn.Conv2d(
            in_channels=w.size(1) * self.c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchEmbed(BaseModule):
    """Patch Embedding for Vision Transformer.

    Adapted from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py

    Different from `mmcv.cnn.bricks.transformer.PatchEmbed`, this module use
    Conv2d and BatchNorm2d to implement PatchEmbedding, and output shape is
    (N, C, H, W).

    Args:
        in_channels (int): The number of input channels.
        embed_dim (int): The embedding dimension.
        resolution (Tuple[int, int]): The resolution of the input feature.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 embed_dim,
                 resolution,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        img_size: Tuple[int, int] = resolution
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.seq = nn.Sequential(
            ConvBN2d(
                in_channels,
                embed_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1),
            build_activation_layer(act_cfg),
            ConvBN2d(
                embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.seq(x)


class PatchMerging(nn.Module):
    """Patch Merging for TinyViT.

    Adapted from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py

    Different from `mmpretrain.models.utils.PatchMerging`, this module use
    Conv2d and BatchNorm2d to implement PatchMerging.

    Args:
        in_channels (int): The number of input channels.
        resolution (Tuple[int, int]): The resolution of the input feature.
        out_channels (int): The number of output channels.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 resolution,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU')):
        super().__init__()

        self.img_size = resolution

        self.act = build_activation_layer(act_cfg)
        self.conv1 = ConvBN2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBN2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels)
        self.conv3 = ConvBN2d(out_channels, out_channels, kernel_size=1)
        self.out_resolution = (resolution[0] // 2, resolution[1] // 2)

    def forward(self, x):
        if len(x.shape) == 3:
            H, W = self.img_size
            B = x.shape[0]
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block for TinyViT. Adapted from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        expand_ratio (int): The expand ratio of the hidden channels.
        drop_rate (float): The drop rate of the block.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 drop_path,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.in_channels = in_channels
        hidden_channels = int(in_channels * expand_ratio)

        # linear
        self.conv1 = ConvBN2d(in_channels, hidden_channels, kernel_size=1)
        self.act = build_activation_layer(act_cfg)
        # depthwise conv
        self.conv2 = ConvBN2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels)
        # linear
        self.conv3 = ConvBN2d(
            hidden_channels, out_channels, kernel_size=1, bn_weight_init=0.0)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act(x)

        return x


class ConvStage(BaseModule):
    """Convolution Stage for TinyViT.

    Adapted from
    https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py

    Args:
        in_channels (int): The number of input channels.
        resolution (Tuple[int, int]): The resolution of the input feature.
        depth (int): The number of blocks in the stage.
        act_cfg (dict): The activation config of the module.
        drop_path (float): The drop path of the block.
        downsample (None | nn.Module): The downsample operation.
            Default: None.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        out_channels (int): The number of output channels.
        conv_expand_ratio (int): The expand ratio of the hidden channels.
            Default: 4.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 resolution,
                 depth,
                 act_cfg,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False,
                 out_channels=None,
                 conv_expand_ratio=4.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = ModuleList([
            MBConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=conv_expand_ratio,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                resolution=resolution,
                in_channels=in_channels,
                out_channels=out_channels,
                act_cfg=act_cfg)
            self.resolution = self.downsample.out_resolution
        else:
            self.downsample = None
            self.resolution = resolution

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MLP(BaseModule):
    """MLP module for TinyViT.

    Args:
        in_channels (int): The number of input channels.
        hidden_channels (int, optional): The number of hidden channels.
            Default: None.
        out_channels (int, optional): The number of output channels.
            Default: None.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
        drop (float): Probability of an element to be zeroed.
            Default: 0.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TinyViTBlock(BaseModule):
    """TinViT Block.

    Args:
        in_channels (int): The number of input channels.
        resolution (Tuple[int, int]): The resolution of the input feature.
        num_heads (int): The number of heads in the multi-head attention.
        window_size (int): The size of the window.
            Default: 7.
        mlp_ratio (float): The ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop (float): Probability of an element to be zeroed.
            Default: 0.
        drop_path (float): The drop path of the block.
            Default: 0.
        local_conv_size (int): The size of the local convolution.
            Default: 3.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 local_conv_size=3,
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert in_channels % num_heads == 0, \
            'dim must be divisible by num_heads'
        head_dim = in_channels // num_heads

        window_resolution = (window_size, window_size)
        self.attn = LeAttention(
            in_channels,
            head_dim,
            num_heads,
            attn_ratio=1,
            resolution=window_resolution)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        self.local_conv = ConvBN2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=local_conv_size,
            stride=1,
            padding=local_conv_size // 2,
            groups=in_channels)

    def forward(self, x):
        H, W = self.img_size
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size -
                     H % self.window_size) % self.window_size
            pad_r = (self.window_size -
                     W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size,
                       C).transpose(2, 3).reshape(
                           B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x


class BasicStage(BaseModule):
    """Basic Stage for TinyViT.

    Args:
        in_channels (int): The number of input channels.
        resolution (Tuple[int, int]): The resolution of the input feature.
        depth (int): The number of blocks in the stage.
        num_heads (int): The number of heads in the multi-head attention.
        window_size (int): The size of the window.
        mlp_ratio (float): The ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop (float): Probability of an element to be zeroed.
            Default: 0.
        drop_path (float): The drop path of the block.
            Default: 0.
        downsample (None | nn.Module): The downsample operation.
            Default: None.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 downsample=None,
                 use_checkpoint=False,
                 local_conv_size=3,
                 out_channels=None,
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = ModuleList([
            TinyViTBlock(
                in_channels=in_channels,
                resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                local_conv_size=local_conv_size,
                act_cfg=act_cfg,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])

        # build patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                resolution=resolution,
                in_channels=in_channels,
                out_channels=out_channels,
                act_cfg=act_cfg)
            self.resolution = self.downsample.out_resolution
        else:
            self.downsample = None
            self.resolution = resolution

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


@MODELS.register_module()
class TinyViT(BaseBackbone):
    """TinyViT.
    A PyTorch implementation of : `TinyViT: Fast Pretraining Distillation
    for Small Vision Transformers<https://arxiv.org/abs/2201.03545v1>`_

    Inspiration from
    https://github.com/microsoft/Cream/blob/main/TinyViT

    Args:
        arch (str | dict): The architecture of TinyViT.
            Default: '5m'.
        img_size (tuple | int): The resolution of the input image.
            Default: (224, 224)
        window_size (list): The size of the window.
            Default: [7, 7, 14, 7]
        in_channels (int): The number of input channels.
            Default: 3.
        depths (list[int]): The depth of each stage.
            Default: [2, 2, 6, 2].
        mlp_ratio (list[int]): The ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_rate (float): Probability of an element to be zeroed.
            Default: 0.
        drop_path_rate (float): The drop path of the block.
            Default: 0.1.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False.
        mbconv_expand_ratio (int): The expand ratio of the mbconv.
            Default: 4.0
        local_conv_size (int): The size of the local conv.
            Default: 3.
        layer_lr_decay (float): The layer lr decay.
            Default: 1.0
        out_indices (int | list[int]): Output from which stages.
            Default: -1
        frozen_stages (int | list[int]): Stages to be frozen (all param fixed).
            Default: -0
        gap_before_final_nrom (bool): Whether to add a gap before the final
            norm. Default: True.
        act_cfg (dict): The activation config of the module.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Default: None.
    """
    arch_settings = {
        '5m': {
            'channels': [64, 128, 160, 320],
            'num_heads': [2, 4, 5, 10],
            'depths': [2, 2, 6, 2],
        },
        '11m': {
            'channels': [64, 128, 256, 448],
            'num_heads': [2, 4, 8, 14],
            'depths': [2, 2, 6, 2],
        },
        '21m': {
            'channels': [96, 192, 384, 576],
            'num_heads': [3, 6, 12, 18],
            'depths': [2, 2, 6, 2],
        },
    }

    def __init__(self,
                 arch='5m',
                 img_size=(224, 224),
                 window_size=[7, 7, 14, 7],
                 in_channels=3,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavaiable arch, please choose from ' \
                f'({set(self.arch_settings)} or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'channels' in arch and 'num_heads' in arch and \
                'depths' in arch, 'The arch dict must have' \
                f'"channels", "num_heads", "window_sizes" ' \
                f'keys, but got {arch.keys()}'

        self.channels = arch['channels']
        self.num_heads = arch['num_heads']
        self.widow_sizes = window_size
        self.img_size = img_size
        self.depths = arch['depths']

        self.num_stages = len(self.channels)

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
        self.layer_lr_decay = layer_lr_decay

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=self.channels[0],
            resolution=self.img_size,
            act_cfg=dict(type='GELU'))
        patches_resolution = self.patch_embed.patches_resolution

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]

        # build stages
        self.stages = ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channel = self.channels[i]
            curr_resolution = (patches_resolution[0] // (2**i),
                               patches_resolution[1] // (2**i))
            drop_path = dpr[sum(self.depths[:i]):sum(self.depths[:i + 1])]
            downsample = PatchMerging if (i < self.num_stages - 1) else None
            out_channels = self.channels[min(i + 1, self.num_stages - 1)]
            if i >= 1:
                stage = BasicStage(
                    in_channels=channel,
                    resolution=curr_resolution,
                    depth=depth,
                    num_heads=self.num_heads[i],
                    window_size=self.widow_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=drop_path,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                    local_conv_size=local_conv_size,
                    out_channels=out_channels,
                    act_cfg=act_cfg)
            else:
                stage = ConvStage(
                    in_channels=channel,
                    resolution=curr_resolution,
                    depth=depth,
                    act_cfg=act_cfg,
                    drop_path=drop_path,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                    out_channels=out_channels,
                    conv_expand_ratio=mbconv_expand_ratio)
            self.stages.append(stage)

            # add output norm
            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, out_channels)[1]
                self.add_module(f'norm{i}', norm_layer)

    def set_layer_lr_decay(self, layer_lr_decay):
        # TODO: add layer_lr_decay
        pass

    def forward(self, x):
        outs = []
        x = self.patch_embed(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean(1)
                    outs.append(norm_layer(gap))
                else:
                    out = norm_layer(x)
                    # convert the (B,L,C) format into (B,C,H,W) format
                    # which would be better for the downstream tasks.
                    B, L, C = out.shape
                    out = out.view(B, *stage.resolution, C)
                    outs.append(out.permute(0, 3, 1, 2))

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(TinyViT, self).train(mode)
        self._freeze_stages()
