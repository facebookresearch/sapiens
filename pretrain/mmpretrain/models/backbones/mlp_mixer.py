# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from ..utils import to_2tuple
from .base_backbone import BaseBackbone


class MixerBlock(BaseModule):
    """Mlp-Mixer basic block.

    Basic module of `MLP-Mixer: An all-MLP Architecture for Vision
    <https://arxiv.org/pdf/2105.01601.pdf>`_

    Args:
        num_tokens (int): The number of patched tokens
        embed_dims (int): The feature dimension
        tokens_mlp_dims (int): The hidden dimension for tokens FFNs
        channels_mlp_dims (int): The hidden dimension for channels FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_tokens,
                 embed_dims,
                 tokens_mlp_dims,
                 channels_mlp_dims,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(MixerBlock, self).__init__(init_cfg=init_cfg)

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.token_mix = FFN(
            embed_dims=num_tokens,
            feedforward_channels=tokens_mlp_dims,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.channel_mix = FFN(
            embed_dims=embed_dims,
            feedforward_channels=channels_mlp_dims,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(MixerBlock, self).init_weights()
        for m in self.token_mix.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)
        for m in self.channel_mix.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        out = self.norm1(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        x = self.channel_mix(self.norm2(x), identity=x)
        return x


@MODELS.register_module()
class MlpMixer(BaseBackbone):
    """Mlp-Mixer backbone.

    Pytorch implementation of `MLP-Mixer: An all-MLP Architecture for Vision
    <https://arxiv.org/pdf/2105.01601.pdf>`_

    Args:
        arch (str | dict): MLP Mixer architecture. If use string, choose from
            'small', 'base' and 'large'. If use dict, it should have below
            keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of MLP blocks.
            - **tokens_mlp_dims** (int): The hidden dimensions for tokens FFNs.
            - **channels_mlp_dims** (int): The The hidden dimensions for
              channels FFNs.

            Defaults to 'base'.
        img_size (int | tuple): The input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        out_indices (Sequence | int): Output from which layer.
            Defaults to -1, means the last layer.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The activation config for FFNs. Default GELU.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each mixer block layer.
            Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 512,
                'num_layers': 8,
                'tokens_mlp_dims': 256,
                'channels_mlp_dims': 2048,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'tokens_mlp_dims': 384,
                'channels_mlp_dims': 3072,
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'tokens_mlp_dims': 512,
                'channels_mlp_dims': 4096,
            }),
    }

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(MlpMixer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'tokens_mlp_dims',
                'channels_mlp_dims'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.tokens_mlp_dims = self.arch_settings['tokens_mlp_dims']
        self.channels_mlp_dims = self.arch_settings['channels_mlp_dims']

        self.img_size = to_2tuple(img_size)

        _patch_cfg = dict(
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must be a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
            else:
                assert index >= self.num_layers, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                num_tokens=num_patches,
                embed_dims=self.embed_dims,
                tokens_mlp_dims=self.tokens_mlp_dims,
                channels_mlp_dims=self.channels_mlp_dims,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(MixerBlock(**_layer_cfg))

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        assert x.shape[2:] == self.img_size, \
            "The MLP-Mixer doesn't support dynamic input shape. " \
            f'Please input images with shape {self.img_size}'
        x, _ = self.patch_embed(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1:
                x = self.norm1(x)

            if i in self.out_indices:
                out = x.transpose(1, 2)
                outs.append(out)

        return tuple(outs)
