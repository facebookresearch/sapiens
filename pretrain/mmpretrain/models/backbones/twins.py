# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.registry import MODELS
from ..utils import ConditionalPositionEncoding, MultiheadAttention


class GlobalSubsampledAttention(MultiheadAttention):
    """Global Sub-sampled Attention (GSA) module.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        sr_ratio (float): The ratio of spatial reduction in attention modules.
            Defaults to 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 norm_cfg=dict(type='LN'),
                 qkv_bias=True,
                 sr_ratio=1,
                 **kwargs):
        super(GlobalSubsampledAttention,
              self).__init__(embed_dims, num_heads, **kwargs)

        self.qkv_bias = qkv_bias
        self.q = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(self.input_dims, embed_dims * 2, bias=qkv_bias)

        # remove self.qkv, here split into self.q, self.kv
        delattr(self, 'qkv')

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # use a conv as the spatial-reduction operation, the kernel_size
            # and stride in conv are equal to the sr_ratio.
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        assert H * W == N, 'The product of h and w of hw_shape must be N, ' \
                           'which is the 2nd dim number of the input Tensor x.'

        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, *hw_shape)  # BNC_2_BCHW
            x = self.sr(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BCHW_2_BNC
            x = self.norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                self.head_dims).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class GSAEncoderLayer(BaseModule):
    """Implements one encoder layer with GlobalSubsampledAttention(GSA).

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (float): The ratio of spatial reduction in attention modules.
            Defaults to 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1.,
                 init_cfg=None):
        super(GSAEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = GlobalSubsampledAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class LocallyGroupedSelfAttention(BaseModule):
    """Locally-grouped Self Attention (LSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        window_size(int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 window_size=1,
                 init_cfg=None):
        super(LocallyGroupedSelfAttention, self).__init__(init_cfg=init_cfg)

        assert embed_dims % num_heads == 0, \
            f'dim {embed_dims} should be divided by num_heads {num_heads}'

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.window_size = window_size

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # calculate attention mask for LSA
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        # [B, _h, _w, window_size, window_size, C]
        x = x.reshape(B, _h, self.window_size, _w, self.window_size,
                      C).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w,
                            self.window_size).transpose(2, 3).reshape(
                                1, _h * _w,
                                self.window_size * self.window_size)
        # [1, _h*_w, window_size*window_size, window_size*window_size]
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-1000.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        # [3, B, _w*_h, nhead, window_size*window_size, dim]
        qkv = self.qkv(x).reshape(B, _h * _w,
                                  self.window_size * self.window_size, 3,
                                  self.num_heads, C // self.num_heads).permute(
                                      3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, _h*_w, n_head, window_size*window_size, window_size*window_size]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.window_size,
                                                  self.window_size, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.window_size,
                                         _w * self.window_size, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LSAEncoderLayer(BaseModule):
    """Implements one encoder layer with LocallyGroupedSelfAttention(LSA).

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
           Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        window_size (int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 qk_scale=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=1,
                 init_cfg=None):

        super(LSAEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = LocallyGroupedSelfAttention(embed_dims, num_heads,
                                                qkv_bias, qk_scale,
                                                attn_drop_rate, drop_rate,
                                                window_size)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


@MODELS.register_module()
class PCPVT(BaseModule):
    """The backbone of Twins-PCPVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        arch (dict, str): PCPVT architecture, a str value in arch zoo or a
            detailed configuration dict with 7 keys, and the length of all the
            values in dict should be the same:

            - depths (List[int]): The number of encoder layers in each stage.
            - embed_dims (List[int]): Embedding dimension in each stage.
            - patch_sizes (List[int]): The patch sizes in each stage.
            - num_heads (List[int]): Numbers of attention head in each stage.
            - strides (List[int]): The strides in each stage.
            - mlp_ratios (List[int]): The ratios of mlp in each stage.
            - sr_ratios (List[int]): The ratios of GSA-encoder layers in each
              stage.

        in_channels (int): Number of input channels. Defaults to 3.
        out_indices (tuple[int]): Output from which stages.
            Defaults to ``(3, )``.
        qkv_bias (bool): Enable bias for qkv if True. Defaults to False.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Defaults to 0.0
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        norm_after_stage(bool, List[bool]): Add extra norm after each stage.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import PCPVT
        >>> import torch
        >>> pcpvt_cfg = {'arch': "small",
        >>>              'norm_after_stage': [False, False, False, True]}
        >>> model = PCPVT(**pcpvt_cfg)
        >>> x = torch.rand(1, 3, 224, 224)
        >>> outputs = model(x)
        >>> print(outputs[-1].shape)
        torch.Size([1, 512, 7, 7])
        >>> pcpvt_cfg['norm_after_stage'] = [True, True, True, True]
        >>> pcpvt_cfg['out_indices'] = (0, 1, 2, 3)
        >>> model = PCPVT(**pcpvt_cfg)
        >>> outputs = model(x)
        >>> for feat in outputs:
        >>>     print(feat.shape)
        torch.Size([1, 64, 56, 56])
        torch.Size([1, 128, 28, 28])
        torch.Size([1, 320, 14, 14])
        torch.Size([1, 512, 7, 7])
    """
    arch_zoo = {
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims':    [64, 128, 320, 512],
                         'depths':        [3, 4, 6, 3],
                         'num_heads':     [1, 2, 5, 8],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [8, 8, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims':    [64, 128, 320, 512],
                         'depths':        [3, 4, 18, 3],
                         'num_heads':     [1, 2, 5, 8],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [8, 8, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims':    [64, 128, 320, 512],
                         'depths':        [3, 8, 27, 3],
                         'num_heads':     [1, 2, 5, 8],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [8, 8, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1]}),
    }   # yapf: disable

    essential_keys = {
        'embed_dims', 'depths', 'num_heads', 'patch_sizes', 'strides',
        'mlp_ratios', 'sr_ratios'
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 out_indices=(3, ),
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 norm_after_stage=False,
                 init_cfg=None):
        super(PCPVT, self).__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            assert isinstance(arch, dict) and (
                set(arch) == self.essential_keys
            ), f'Custom arch needs a dict with keys {self.essential_keys}.'
            self.arch_settings = arch

        self.depths = self.arch_settings['depths']
        self.embed_dims = self.arch_settings['embed_dims']
        self.patch_sizes = self.arch_settings['patch_sizes']
        self.strides = self.arch_settings['strides']
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.num_heads = self.arch_settings['num_heads']
        self.sr_ratios = self.arch_settings['sr_ratios']

        self.num_extra_tokens = 0  # there is no cls-token in Twins
        self.num_stage = len(self.depths)
        for key, value in self.arch_settings.items():
            assert isinstance(value, list) and len(value) == self.num_stage, (
                'Length of setting item in arch dict must be type of list and'
                ' have the same length.')

        # patch_embeds
        self.patch_embeds = ModuleList()
        self.position_encoding_drops = ModuleList()
        self.stages = ModuleList()

        for i in range(self.num_stage):
            # use in_channels of the model in the first stage
            if i == 0:
                stage_in_channels = in_channels
            else:
                stage_in_channels = self.embed_dims[i - 1]

            self.patch_embeds.append(
                PatchEmbed(
                    in_channels=stage_in_channels,
                    embed_dims=self.embed_dims[i],
                    conv_type='Conv2d',
                    kernel_size=self.patch_sizes[i],
                    stride=self.strides[i],
                    padding='corner',
                    norm_cfg=dict(type='LN')))

            self.position_encoding_drops.append(nn.Dropout(p=drop_rate))

        # PEGs
        self.position_encodings = ModuleList([
            ConditionalPositionEncoding(embed_dim, embed_dim)
            for embed_dim in self.embed_dims
        ])

        # stochastic depth
        total_depth = sum(self.depths)
        self.dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        cur = 0

        for k in range(len(self.depths)):
            _block = ModuleList([
                GSAEncoderLayer(
                    embed_dims=self.embed_dims[k],
                    num_heads=self.num_heads[k],
                    feedforward_channels=self.mlp_ratios[k] *
                    self.embed_dims[k],
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=self.dpr[cur + i],
                    num_fcs=2,
                    qkv_bias=qkv_bias,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=norm_cfg,
                    sr_ratio=self.sr_ratios[k]) for i in range(self.depths[k])
            ])
            self.stages.append(_block)
            cur += self.depths[k]

        self.out_indices = out_indices

        assert isinstance(norm_after_stage, (bool, list))
        if isinstance(norm_after_stage, bool):
            self.norm_after_stage = [norm_after_stage] * self.num_stage
        else:
            self.norm_after_stage = norm_after_stage
        assert len(self.norm_after_stage) == self.num_stage, \
            (f'Number of norm_after_stage({len(self.norm_after_stage)}) should'
             f' be equal to the number of stages({self.num_stage}).')

        for i, has_norm in enumerate(self.norm_after_stage):
            assert isinstance(has_norm, bool), 'norm_after_stage should be ' \
                                               'bool or List[bool].'
            if has_norm and norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, self.embed_dims[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm_after_stage{i}', norm_layer)

    def init_weights(self):
        if self.init_cfg is not None:
            super(PCPVT, self).init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        outputs = list()

        b = x.shape[0]

        for i in range(self.num_stage):
            x, hw_shape = self.patch_embeds[i](x)
            h, w = hw_shape
            x = self.position_encoding_drops[i](x)
            for j, blk in enumerate(self.stages[i]):
                x = blk(x, hw_shape)
                if j == 0:
                    x = self.position_encodings[i](x, hw_shape)

            norm_layer = getattr(self, f'norm_after_stage{i}')
            x = norm_layer(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)


@MODELS.register_module()
class SVT(PCPVT):
    """The backbone of Twins-SVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        arch (dict, str): SVT architecture, a str value in arch zoo or a
            detailed configuration dict with 8 keys, and the length of all the
            values in dict should be the same:

            - depths (List[int]): The number of encoder layers in each stage.
            - embed_dims (List[int]): Embedding dimension in each stage.
            - patch_sizes (List[int]): The patch sizes in each stage.
            - num_heads (List[int]): Numbers of attention head in each stage.
            - strides (List[int]): The strides in each stage.
            - mlp_ratios (List[int]): The ratios of mlp in each stage.
            - sr_ratios (List[int]): The ratios of GSA-encoder layers in each
              stage.
            - windiow_sizes (List[int]): The window sizes in LSA-encoder layers
              in each stage.

        in_channels (int): Number of input channels. Defaults to 3.
        out_indices (tuple[int]): Output from which stages.
            Defaults to (3, ).
        qkv_bias (bool): Enable bias for qkv if True. Defaults to False.
        drop_rate (float): Dropout rate. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.0
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.2.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        norm_after_stage(bool, List[bool]): Add extra norm after each stage.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SVT
        >>> import torch
        >>> svt_cfg = {'arch': "small",
        >>>            'norm_after_stage': [False, False, False, True]}
        >>> model = SVT(**svt_cfg)
        >>> x = torch.rand(1, 3, 224, 224)
        >>> outputs = model(x)
        >>> print(outputs[-1].shape)
        torch.Size([1, 512, 7, 7])
        >>> svt_cfg["out_indices"] = (0, 1, 2, 3)
        >>> svt_cfg["norm_after_stage"] = [True, True, True, True]
        >>> model = SVT(**svt_cfg)
        >>> output = model(x)
        >>> for feat in output:
        >>>     print(feat.shape)
        torch.Size([1, 64, 56, 56])
        torch.Size([1, 128, 28, 28])
        torch.Size([1, 320, 14, 14])
        torch.Size([1, 512, 7, 7])
    """
    arch_zoo = {
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims':    [64, 128, 256, 512],
                         'depths':        [2, 2, 10, 4],
                         'num_heads':     [2, 4, 8, 16],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [4, 4, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1],
                         'window_sizes':  [7, 7, 7, 7]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims':    [96, 192, 384, 768],
                         'depths':        [2, 2, 18, 2],
                         'num_heads':     [3, 6, 12, 24],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [4, 4, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1],
                         'window_sizes':  [7, 7, 7, 7]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims':    [128, 256, 512, 1024],
                         'depths':        [2, 2, 18, 2],
                         'num_heads':     [4, 8, 16, 32],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [4, 4, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1],
                         'window_sizes':  [7, 7, 7, 7]}),
    }  # yapf: disable

    essential_keys = {
        'embed_dims', 'depths', 'num_heads', 'patch_sizes', 'strides',
        'mlp_ratios', 'sr_ratios', 'window_sizes'
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 out_indices=(3, ),
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.0,
                 norm_cfg=dict(type='LN'),
                 norm_after_stage=False,
                 init_cfg=None):
        super(SVT, self).__init__(arch, in_channels, out_indices, qkv_bias,
                                  drop_rate, attn_drop_rate, drop_path_rate,
                                  norm_cfg, norm_after_stage, init_cfg)

        self.window_sizes = self.arch_settings['window_sizes']

        for k in range(self.num_stage):
            for i in range(self.depths[k]):
                # in even-numbered layers of each stage, replace GSA with LSA
                if i % 2 == 0:
                    ffn_channels = self.mlp_ratios[k] * self.embed_dims[k]
                    self.stages[k][i] = \
                        LSAEncoderLayer(
                            embed_dims=self.embed_dims[k],
                            num_heads=self.num_heads[k],
                            feedforward_channels=ffn_channels,
                            drop_rate=drop_rate,
                            norm_cfg=norm_cfg,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=self.dpr[sum(self.depths[:k])+i],
                            qkv_bias=qkv_bias,
                            window_size=self.window_sizes[k])
