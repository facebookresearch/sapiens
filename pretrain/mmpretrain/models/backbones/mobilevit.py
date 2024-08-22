# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from torch import nn

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from .mobilenet_v2 import InvertedResidual
from .vision_transformer import TransformerEncoderLayer


class MobileVitBlock(nn.Module):
    """MobileViT block.

    According to the paper, the MobileViT block has a local representation.
    a transformer-as-convolution layer which consists of a global
    representation with unfolding and folding, and a final fusion layer.

    Args:
        in_channels (int): Number of input image channels.
        transformer_dim (int): Number of transformer channels.
        ffn_dim (int): Number of ffn channels in transformer block.
        out_channels (int): Number of channels in output.
        conv_ksize (int): Conv kernel size in local representation
            and fusion. Defaults to 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Defaults to dict(type='Swish').
        num_transformer_blocks (int): Number of transformer blocks in
            a MobileViT block. Defaults to 2.
        patch_size (int): Patch size for unfolding and folding.
             Defaults to 2.
        num_heads (int): Number of heads in global representation.
             Defaults to 4.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        no_fusion (bool): Whether to remove the fusion layer.
            Defaults to False.
        transformer_norm_cfg (dict, optional): Config dict for normalization
            layer in transformer. Defaults to dict(type='LN').
    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            out_channels: int,
            conv_ksize: int = 3,
            conv_cfg: Optional[dict] = None,
            norm_cfg: Optional[dict] = dict(type='BN'),
            act_cfg: Optional[dict] = dict(type='Swish'),
            num_transformer_blocks: int = 2,
            patch_size: int = 2,
            num_heads: int = 4,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            no_fusion: bool = False,
            transformer_norm_cfg: Callable = dict(type='LN'),
    ):
        super(MobileVitBlock, self).__init__()

        self.local_rep = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=conv_ksize,
                padding=int((conv_ksize - 1) / 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=in_channels,
                out_channels=transformer_dim,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None),
        )

        global_rep = [
            TransformerEncoderLayer(
                embed_dims=transformer_dim,
                num_heads=num_heads,
                feedforward_channels=ffn_dim,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=True,
                act_cfg=dict(type='Swish'),
                norm_cfg=transformer_norm_cfg)
            for _ in range(num_transformer_blocks)
        ]
        global_rep.append(
            build_norm_layer(transformer_norm_cfg, transformer_dim)[1])
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = ConvModule(
            in_channels=transformer_dim,
            out_channels=out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = ConvModule(
                in_channels=in_channels + out_channels,
                out_channels=out_channels,
                kernel_size=conv_ksize,
                padding=int((conv_ksize - 1) / 2),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        self.patch_size = (patch_size, patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.local_rep(x)

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(
            W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w # noqa
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function. # noqa
            x = F.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w,
                      patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w # noqa
        x = x.reshape(B, C, num_patches,
                      self.patch_area).transpose(1, 3).reshape(
                          B * self.patch_area, num_patches, -1)

        # Global representations
        x = self.global_rep(x)

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w,
                                      patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W] # noqa
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h,
                                      num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(
                x, size=(H, W), mode='bilinear', align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


@MODELS.register_module()
class MobileViT(BaseBackbone):
    """MobileViT backbone.

    A PyTorch implementation of : `MobileViT: Light-weight, General-purpose,
    and Mobile-friendly Vision Transformer <https://arxiv.org/pdf/2110.02178.pdf>`_

    Modified from the `official repo
    <https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilevit.py>`_.

    Args:
        arch (str | List[list]): Architecture of MobileViT.

            - If a string, choose from "small", "x_small" and "xx_small".

            - If a list, every item should be also a list, and the first item
              of the sub-list can be chosen from "moblienetv2" and "mobilevit",
              which indicates the type of this layer sequence. If "mobilenetv2",
              the other items are the arguments of :attr:`~MobileViT.make_mobilenetv2_layer`
              (except ``in_channels``) and if "mobilevit", the other items are
              the arguments of :attr:`~MobileViT.make_mobilevit_layer`
              (except ``in_channels``).

            Defaults to "small".
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int): Channels of stem layer.  Defaults to 16.
        last_exp_factor (int): Channels expand factor of last layer.
            Defaults to 4.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (4, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Defaults to dict(type='Swish').
        init_cfg (dict, optional): Initialization config dict.
    """  # noqa

    # Parameters to build layers. The first param is the type of layer.
    # For `mobilenetv2` layer, the rest params from left to right are:
    #     out channels, stride, num of blocks, expand_ratio.
    # For `mobilevit` layer, the rest params from left to right are:
    #     out channels, stride, transformer_channels, ffn channels,
    # num of transformer blocks, expand_ratio.
    arch_settings = {
        'small': [
            ['mobilenetv2', 32, 1, 1, 4],
            ['mobilenetv2', 64, 2, 3, 4],
            ['mobilevit', 96, 2, 144, 288, 2, 4],
            ['mobilevit', 128, 2, 192, 384, 4, 4],
            ['mobilevit', 160, 2, 240, 480, 3, 4],
        ],
        'x_small': [
            ['mobilenetv2', 32, 1, 1, 4],
            ['mobilenetv2', 48, 2, 3, 4],
            ['mobilevit', 64, 2, 96, 192, 2, 4],
            ['mobilevit', 80, 2, 120, 240, 4, 4],
            ['mobilevit', 96, 2, 144, 288, 3, 4],
        ],
        'xx_small': [
            ['mobilenetv2', 16, 1, 1, 2],
            ['mobilenetv2', 24, 2, 3, 2],
            ['mobilevit', 48, 2, 64, 128, 2, 2],
            ['mobilevit', 64, 2, 80, 160, 4, 2],
            ['mobilevit', 80, 2, 96, 192, 3, 2],
        ]
    }

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 stem_channels=16,
                 last_exp_factor=4,
                 out_indices=(4, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Swish'),
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MobileViT, self).__init__(init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a list.'
            arch = self.arch_settings[arch]

        self.arch = arch
        self.num_stages = len(arch)

        # check out indices and frozen stages
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        if frozen_stages not in range(-1, self.num_stages):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{self.num_stages}). '
                             f'But received {frozen_stages}')
        self.frozen_stages = frozen_stages

        _make_layer_func = {
            'mobilenetv2': self.make_mobilenetv2_layer,
            'mobilevit': self.make_mobilevit_layer,
        }

        self.stem = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        in_channels = stem_channels
        layers = []
        for i, layer_settings in enumerate(arch):
            layer_type, settings = layer_settings[0], layer_settings[1:]
            layer, out_channels = _make_layer_func[layer_type](in_channels,
                                                               *settings)
            layers.append(layer)
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.conv_1x1_exp = ConvModule(
            in_channels=in_channels,
            out_channels=last_exp_factor * in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    @staticmethod
    def make_mobilevit_layer(in_channels,
                             out_channels,
                             stride,
                             transformer_dim,
                             ffn_dim,
                             num_transformer_blocks,
                             expand_ratio=4):
        """Build mobilevit layer, which consists of one InvertedResidual and
        one MobileVitBlock.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The output channels.
            stride (int): The stride of the first 3x3 convolution in the
                ``InvertedResidual`` layers.
            transformer_dim (int): The channels of the transformer layers.
            ffn_dim (int): The mid-channels of the feedforward network in
                transformer layers.
            num_transformer_blocks (int): The number of transformer blocks.
            expand_ratio (int): adjusts number of channels of the hidden layer
                in ``InvertedResidual`` by this amount. Defaults to 4.
        """
        layer = []
        layer.append(
            InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                act_cfg=dict(type='Swish'),
            ))
        layer.append(
            MobileVitBlock(
                in_channels=out_channels,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                out_channels=out_channels,
                num_transformer_blocks=num_transformer_blocks,
            ))
        return nn.Sequential(*layer), out_channels

    @staticmethod
    def make_mobilenetv2_layer(in_channels,
                               out_channels,
                               stride,
                               num_blocks,
                               expand_ratio=4):
        """Build mobilenetv2 layer, which consists of several InvertedResidual
        layers.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The output channels.
            stride (int): The stride of the first 3x3 convolution in the
                ``InvertedResidual`` layers.
            num_blocks (int): The number of ``InvertedResidual`` blocks.
            expand_ratio (int): adjusts number of channels of the hidden layer
                in ``InvertedResidual`` by this amount. Defaults to 4.
        """
        layer = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1

            layer.append(
                InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    act_cfg=dict(type='Swish'),
                ))
            in_channels = out_channels
        return nn.Sequential(*layer), out_channels

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileViT, self).train(mode)
        self._freeze_stages()

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.conv_1x1_exp(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
