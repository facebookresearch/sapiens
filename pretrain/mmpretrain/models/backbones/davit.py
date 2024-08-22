# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import Conv2d
from mmcv.cnn.bricks.transformer import FFN, AdaptivePadding, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from ..utils import ShiftWindowMSA


class DaViTWindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module for DaViT.

    The differences between DaViTWindowMSA & WindowMSA:
        1. Without relative position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ConvPosEnc(BaseModule):
    """DaViT conv pos encode block.

    Args:
        embed_dims (int): Number of input channels.
        kernel_size (int): The kernel size of the first convolution.
            Defaults to 3.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self, embed_dims, kernel_size=3, init_cfg=None):
        super(ConvPosEnc, self).__init__(init_cfg)
        self.proj = Conv2d(
            embed_dims,
            embed_dims,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=embed_dims)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x


class DaViTDownSample(BaseModule):
    """DaViT down sampole block.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel size of the first convolution.
            Defaults to 2.
        stride (int): The stride of the second convluation module.
            Defaults to 2.
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Defaults to "corner".
        dilation (int): Dilation of the convolution layers. Defaults to 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_type='Conv2d',
                 kernel_size=2,
                 stride=2,
                 padding='same',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        else:
            self.norm = None

    def forward(self, x, input_size):
        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        H, W = input_size
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = self.projection(x)
        output_size = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        return x, output_size


class ChannelAttention(BaseModule):
    """DaViT channel attention.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self, embed_dims, num_heads=8, qkv_bias=False, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, x):
        B, N, _ = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)

        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        return x


class ChannelBlock(BaseModule):
    """DaViT channel attention block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 drop_path=0.,
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.with_cp = with_cp

        self.cpe1 = ConvPosEnc(embed_dims=embed_dims, kernel_size=3)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ChannelAttention(
            embed_dims, num_heads=num_heads, qkv_bias=qkv_bias)
        self.cpe2 = ConvPosEnc(embed_dims=embed_dims, kernel_size=3)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            **ffn_cfgs
        }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(**_ffn_cfgs)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            x = self.cpe1(x, hw_shape)
            identity = x
            x = self.norm1(x)
            x = self.attn(x)
            x = x + identity

            x = self.cpe2(x, hw_shape)
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SpatialBlock(BaseModule):
    """DaViT spatial attention block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        attn_cfgs (dict): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 drop_path=0.,
                 pad_small_map=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SpatialBlock, self).__init__(init_cfg)
        self.with_cp = with_cp

        self.cpe1 = ConvPosEnc(embed_dims=embed_dims, kernel_size=3)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'qkv_bias': qkv_bias,
            'pad_small_map': pad_small_map,
            'window_msa': DaViTWindowMSA,
            **attn_cfgs
        }
        self.attn = ShiftWindowMSA(**_attn_cfgs)
        self.cpe2 = ConvPosEnc(embed_dims=embed_dims, kernel_size=3)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            **ffn_cfgs
        }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(**_ffn_cfgs)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            x = self.cpe1(x, hw_shape)
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            x = self.cpe2(x, hw_shape)
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class DaViTBlock(BaseModule):
    """DaViT block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        attn_cfgs (dict): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 drop_path=0.,
                 pad_small_map=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(DaViTBlock, self).__init__(init_cfg)
        self.spatial_block = SpatialBlock(
            embed_dims,
            num_heads,
            window_size=window_size,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            pad_small_map=pad_small_map,
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            norm_cfg=norm_cfg,
            with_cp=with_cp)
        self.channel_block = ChannelBlock(
            embed_dims,
            num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            ffn_cfgs=ffn_cfgs,
            norm_cfg=norm_cfg,
            with_cp=False)

    def forward(self, x, hw_shape):
        x = self.spatial_block(x, hw_shape)
        x = self.channel_block(x, hw_shape)

        return x


class DaViTBlockSequence(BaseModule):
    """Module with successive DaViT blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive DaViT blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        downsample (bool): Downsample the output of blocks by patch merging.
            Defaults to False.
        downsample_cfg (dict): The extra config of the patch merging layer.
            Defaults to empty dict.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=7,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 with_cp=False,
                 pad_small_map=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'num_heads': num_heads,
                'window_size': window_size,
                'ffn_ratio': ffn_ratio,
                'qkv_bias': qkv_bias,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **block_cfgs[i]
            }
            block = DaViTBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': 2 * embed_dims,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = DaViTDownSample(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x, in_shape, do_downsample=True):
        for block in self.blocks:
            x = block(x, in_shape)

        if self.downsample is not None and do_downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.out_channels
        else:
            return self.embed_dims


@MODELS.register_module()
class DaViT(BaseBackbone):
    """DaViT.

    A PyTorch implement of : `DaViT: Dual Attention Vision Transformers
    <https://arxiv.org/abs/2204.03645v1>`_

    Inspiration from
    https://github.com/dingmyu/davit

    Args:
        arch (str | dict): DaViT architecture. If use string, choose from
            'tiny', 'small', 'base' and 'large', 'huge', 'giant'. If use dict,
            it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (List[int]): The number of heads in attention
              modules of each stage.

            Defaults to 't'.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int): The height and width of the window. Defaults to 7.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``
        stage_cfgs (Sequence[dict] | dict): Extra config dict for each
            stage. Defaults to an empty dict.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'], {
                            'embed_dims': 96,
                            'depths': [1, 1, 3, 1],
                            'num_heads': [3, 6, 12, 24]
                        }),
        **dict.fromkeys(['s', 'small'], {
                            'embed_dims': 96,
                            'depths': [1, 1, 9, 1],
                            'num_heads': [3, 6, 12, 24]
                        }),
        **dict.fromkeys(['b', 'base'], {
                            'embed_dims': 128,
                            'depths': [1, 1, 9, 1],
                            'num_heads': [4, 8, 16, 32]
                        }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 192,
                'depths': [1, 1, 9, 1],
                'num_heads': [6, 12, 24, 48]
            }),
        **dict.fromkeys(
            ['h', 'huge'], {
                'embed_dims': 256,
                'depths': [1, 1, 9, 1],
                'num_heads': [8, 16, 32, 64]
            }),
        **dict.fromkeys(
            ['g', 'giant'], {
                'embed_dims': 384,
                'depths': [1, 1, 12, 3],
                'num_heads': [12, 24, 48, 96]
            }),
    }

    def __init__(self,
                 arch='t',
                 patch_size=4,
                 in_channels=3,
                 window_size=7,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 drop_path_rate=0.1,
                 out_after_downsample=False,
                 pad_small_map=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 frozen_stages=-1,
                 norm_eval=False,
                 out_indices=(3, ),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.out_after_downsample = out_after_downsample
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # stochastic depth decay rule
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        _patch_cfg = dict(
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=7,
            stride=patch_size,
            padding='same',
            norm_cfg=dict(type='LN'),
        )
        self.patch_embed = PatchEmbed(**_patch_cfg)

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'ffn_ratio': ffn_ratio,
                'qkv_bias': qkv_bias,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                **stage_cfg
            }

            stage = DaViTBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        self.num_features = embed_dims[:-1]

        # add a norm layer for each output
        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(
                x, hw_shape, do_downsample=self.out_after_downsample)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if stage.downsample is not None and not self.out_after_downsample:
                x, hw_shape = stage.downsample(x, hw_shape)

        return tuple(outs)
