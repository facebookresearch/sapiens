# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from ..utils import (RotaryEmbeddingFast, SwiGLUFFN, build_norm_layer,
                     resize_pos_embed)
from .vision_transformer import VisionTransformer


class AttentionWithRoPE(BaseModule):
    """Multi-head Attention Module with 2D sincos position embedding (RoPE).

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): If True, add a learnable bias to q and v. Note
            that we follows the official implementation where ``k_bias``
            is 0. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        rope (:obj:`torch.nn.Module`, optional): If it is an object of the
            ``RotaryEmbedding``, the rotation of the token position will be
            performed before the softmax. Defaults to None.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 rope=None,
                 with_cls_token=True,
                 init_cfg=None):
        super(AttentionWithRoPE, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.with_cls_token = with_cls_token

        self.rope = rope

    def forward(self, x, patch_resolution, mask=None, ids_restore=None, ids_keep=None):
        B, N, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self.rope:
            if self.with_cls_token:
                q_t = q[:, :, 1:, :]
                ro_q_t = self.rope(q_t, patch_resolution, mask, ids_restore, ids_keep)
                q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

                k_t = k[:, :, 1:, :] if self.with_cls_token else k
                ro_k_t = self.rope(k_t, patch_resolution, mask, ids_restore, ids_keep)
                k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)
            else:
                q = self.rope(q, patch_resolution, mask, ids_restore, ids_keep)
                k = self.rope(k, patch_resolution, mask, ids_restore, ids_keep)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EVA02EndcoderLayer(BaseModule):
    """Implements one encoder EVA02EndcoderLayer in EVA02.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension of FFNs.
        sub_ln (bool): Whether to add the sub layer normalization
            in the attention module. Defaults to False.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool): enable bias for projection in the attention module
            if True. Defaults to True.
        rope (:obj:`torch.nn.Module`, optional): RotaryEmbedding object
            in the attention module. Defaults to None.
        drop_rate (float): Dropout rate in the mlp module. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 sub_ln=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_bias=True,
                 rope=None,
                 with_cls_token=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(EVA02EndcoderLayer, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)

        self.attn = AttentionWithRoPE(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            rope=rope,
            with_cls_token=with_cls_token)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)

        if drop_rate > 0:
            dropout_layer = dict(type='Dropout', drop_prob=drop_rate)
        else:
            dropout_layer = None

        if sub_ln:
            ffn_norm = norm_cfg
        else:
            ffn_norm = None

        self.mlp = SwiGLUFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            dropout_layer=dropout_layer,
            norm_cfg=ffn_norm,
            add_identity=False,
        )

    def forward(self, x, patch_resolution, mask=None, ids_restore=None, ids_keep=None):
        inputs = x
        x = self.norm1(x)
        x = self.attn(x, patch_resolution, mask, ids_restore, ids_keep)
        x = self.drop_path(x)
        x = inputs + x

        inputs = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = inputs + x

        return x


@MODELS.register_module()
class ViTEVA02(VisionTransformer):
    """EVA02 Vision Transformer.

    A PyTorch implement of : `EVA-02: A Visual Representation for Neon Genesis
    <https://arxiv.org/abs/2303.11331>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'tiny', 'small', 'base', 'large'. If use dict,
            it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **mlp_ratio** (float): The ratio of the mlp module.

            Defaults to 'tiny'.

        sub_ln (bool): Whether to add the sub layer normalization in swiglu.
            Defaults to False.
        drop_rate (float): Probability of an element to be zeroed in the
            mlp module. Defaults to 0.
        attn_drop_rate (float): Probability of an element to be zeroed after
            the softmax in the attention. Defaults to 0.
        proj_drop_rate (float): Probability of an element to be zeroed after
            projection in the attention. Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        **kwargs(dict, optional): Other args for Vision Transformer.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['t', 'ti', 'tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': int(192 * 4 * 2 / 3)
            }),
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': int(384 * 4 * 2 / 3)
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': int(768 * 4 * 2 / 3)
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': int(1024 * 4 * 2 / 3)
            }),
        **dict.fromkeys(
            ['m', 'mammoth'], {
                'embed_dims': 1536,
                'num_layers': 40,
                'num_heads': 24, ## make sure the num_heads divides the embed_dims
                'feedforward_channels': int(1536 * 4 * 2 / 3)
            }),  
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='tiny',
                 sub_ln=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 use_rope=True,
                 norm_cfg=dict(type='LN'),
                 with_cls_token=True,
                 layer_cfgs=dict(),
                 **kwargs):
        # set essential args for Vision Transformer
        kwargs.update(
            arch=arch,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            with_cls_token=with_cls_token)
        super(ViTEVA02, self).__init__(**kwargs)

        self.num_heads = self.arch_settings['num_heads']

        # Set RoPE
        head_dim = self.embed_dims // self.num_heads

        if use_rope == True:
            self.rope = RotaryEmbeddingFast(
                embed_dims=head_dim, patch_resolution=self.patch_resolution)
        else:
            self.rope = None

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                sub_ln=sub_ln,
                norm_cfg=norm_cfg,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_rate=drop_rate,
                qkv_bias=qkv_bias,
                rope=self.rope,
                with_cls_token=with_cls_token,
                drop_path_rate=dpr[i])
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(EVA02EndcoderLayer(**_layer_cfg))

    def forward(self, x, mask=None, ids_restore=None, ids_keep=None):
        B = x.shape[0]

        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, patch_resolution, mask, ids_restore, ids_keep)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)
