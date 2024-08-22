# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from ..utils import to_2tuple
from .base_backbone import BaseBackbone


class TransformerBlock(BaseModule):
    """Implement a transformer block in TnTLayer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        ffn_ratio (int): A ratio to calculate the hidden_dims in ffn layer.
            Default: 4
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        num_fcs (int): The number of fully-connected layers for FFNs. Default 2
        qkv_bias (bool): Enable bias for qkv if True. Default False
        act_cfg (dict): The activation config for FFNs. Defaults to GELU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim).
            (batch, n, embed_dim) is common case in CV.  Defaults to False
        init_cfg (dict, optional): Initialization config dict. Defaults to None
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 init_cfg=None):
        super(TransformerBlock, self).__init__(init_cfg=init_cfg)

        self.norm_attn = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first)

        self.norm_ffn = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * ffn_ratio,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if not qkv_bias:
            self.attn.attn.in_proj_bias = None

    def forward(self, x):
        x = self.attn(self.norm_attn(x), identity=x)
        x = self.ffn(self.norm_ffn(x), identity=x)
        return x


class TnTLayer(BaseModule):
    """Implement one encoder layer in Transformer in Transformer.

    Args:
        num_pixel (int): The pixel number in target patch transformed with
            a linear projection in inner transformer
        embed_dims_inner (int): Feature dimension in inner transformer block
        embed_dims_outer (int): Feature dimension in outer transformer block
        num_heads_inner (int): Parallel attention heads in inner transformer.
        num_heads_outer (int): Parallel attention heads in outer transformer.
        inner_block_cfg (dict): Extra config of inner transformer block.
            Defaults to empty dict.
        outer_block_cfg (dict): Extra config of outer transformer block.
            Defaults to empty dict.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization
        init_cfg (dict, optional): Initialization config dict. Defaults to None
    """

    def __init__(self,
                 num_pixel,
                 embed_dims_inner,
                 embed_dims_outer,
                 num_heads_inner,
                 num_heads_outer,
                 inner_block_cfg=dict(),
                 outer_block_cfg=dict(),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(TnTLayer, self).__init__(init_cfg=init_cfg)

        self.inner_block = TransformerBlock(
            embed_dims=embed_dims_inner,
            num_heads=num_heads_inner,
            **inner_block_cfg)

        self.norm_proj = build_norm_layer(norm_cfg, embed_dims_inner)[1]
        self.projection = nn.Linear(
            embed_dims_inner * num_pixel, embed_dims_outer, bias=True)

        self.outer_block = TransformerBlock(
            embed_dims=embed_dims_outer,
            num_heads=num_heads_outer,
            **outer_block_cfg)

    def forward(self, pixel_embed, patch_embed):
        pixel_embed = self.inner_block(pixel_embed)

        B, N, C = patch_embed.size()
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.projection(
            self.norm_proj(pixel_embed).reshape(B, N - 1, -1))
        patch_embed = self.outer_block(patch_embed)

        return pixel_embed, patch_embed


class PixelEmbed(BaseModule):
    """Image to Pixel Embedding.

    Args:
        img_size (int | tuple): The size of input image
        patch_size (int): The size of one patch
        in_channels (int): The num of input channels
        embed_dims_inner (int): The num of channels of the target patch
            transformed with a linear projection in inner transformer
        stride (int): The stride of the conv2d layer. We use a conv2d layer
            and a unfold layer to implement image to pixel embedding.
        init_cfg (dict, optional): Initialization config dict
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims_inner=48,
                 stride=4,
                 init_cfg=None):
        super(PixelEmbed, self).__init__(init_cfg=init_cfg)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patches_resolution property necessary for resizing
        # positional embedding
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dims_inner = embed_dims_inner

        new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(
            in_channels,
            self.embed_dims_inner,
            kernel_size=7,
            padding=3,
            stride=stride)
        self.unfold = nn.Unfold(
            kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model " \
            f'({self.img_size[0]}*{self.img_size[1]}).'
        x = self.proj(x)
        x = self.unfold(x)
        x = x.transpose(1,
                        2).reshape(B * self.num_patches, self.embed_dims_inner,
                                   self.new_patch_size[0],
                                   self.new_patch_size[1])
        x = x + pixel_pos
        x = x.reshape(B * self.num_patches, self.embed_dims_inner,
                      -1).transpose(1, 2)
        return x


@MODELS.register_module()
class TNT(BaseBackbone):
    """Transformer in Transformer.

    A PyTorch implement of: `Transformer in Transformer
    <https://arxiv.org/abs/2103.00112>`_

    Inspiration from
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/tnt.py

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size. Defaults to 224
        patch_size (int | tuple): The patch size. Deault to 16
        in_channels (int): Number of input channels. Defaults to 3
        ffn_ratio (int): A ratio to calculate the hidden_dims in ffn layer.
            Default: 4
        qkv_bias (bool): Enable bias for qkv if True. Default False
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.
        drop_path_rate (float): stochastic depth rate. Default 0.
        act_cfg (dict): The activation config for FFNs. Defaults to GELU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization
        first_stride (int): The stride of the conv2d layer. We use a conv2d
            layer and a unfold layer to implement image to pixel embedding.
        num_fcs (int): The number of fully-connected layers for FFNs. Default 2
        init_cfg (dict, optional): Initialization config dict
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims_outer': 384,
                'embed_dims_inner': 24,
                'num_layers': 12,
                'num_heads_outer': 6,
                'num_heads_inner': 4
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims_outer': 640,
                'embed_dims_inner': 40,
                'num_layers': 12,
                'num_heads_outer': 10,
                'num_heads_inner': 4
            })
    }

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 ffn_ratio=4,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 first_stride=4,
                 num_fcs=2,
                 init_cfg=[
                     dict(type='TruncNormal', layer='Linear', std=.02),
                     dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
                 ]):
        super(TNT, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims_outer', 'embed_dims_inner', 'num_layers',
                'num_heads_inner', 'num_heads_outer'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims_inner = self.arch_settings['embed_dims_inner']
        self.embed_dims_outer = self.arch_settings['embed_dims_outer']
        # embed_dims for consistency with other models
        self.embed_dims = self.embed_dims_outer
        self.num_layers = self.arch_settings['num_layers']
        self.num_heads_inner = self.arch_settings['num_heads_inner']
        self.num_heads_outer = self.arch_settings['num_heads_outer']

        self.pixel_embed = PixelEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims_inner=self.embed_dims_inner,
            stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size[0] * new_patch_size[1]

        self.norm1_proj = build_norm_layer(norm_cfg, num_pixel *
                                           self.embed_dims_inner)[1]
        self.projection = nn.Linear(num_pixel * self.embed_dims_inner,
                                    self.embed_dims_outer)
        self.norm2_proj = build_norm_layer(norm_cfg, self.embed_dims_outer)[1]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims_outer))
        self.patch_pos = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dims_outer))
        self.pixel_pos = nn.Parameter(
            torch.zeros(1, self.embed_dims_inner, new_patch_size[0],
                        new_patch_size[1]))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, self.num_layers)
        ]  # stochastic depth decay rule
        self.layers = ModuleList()
        for i in range(self.num_layers):
            block_cfg = dict(
                ffn_ratio=ffn_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                num_fcs=num_fcs,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                batch_first=True)
            self.layers.append(
                TnTLayer(
                    num_pixel=num_pixel,
                    embed_dims_inner=self.embed_dims_inner,
                    embed_dims_outer=self.embed_dims_outer,
                    num_heads_inner=self.num_heads_inner,
                    num_heads_outer=self.num_heads_outer,
                    inner_block_cfg=block_cfg,
                    outer_block_cfg=block_cfg,
                    norm_cfg=norm_cfg))

        self.norm = build_norm_layer(norm_cfg, self.embed_dims_outer)[1]

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.patch_pos, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)

    def forward(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(
            self.projection(
                self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
        patch_embed = torch.cat(
            (self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.drop_after_pos(patch_embed)

        for layer in self.layers:
            pixel_embed, patch_embed = layer(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return (patch_embed[:, 0], )
