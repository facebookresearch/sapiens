# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import (Conv2dAdaptivePadding, build_activation_layer,
                             build_norm_layer)
from mmengine.utils import digit_version

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


@MODELS.register_module()
class ConvMixer(BaseBackbone):
    """ConvMixer.                              .

    A PyTorch implementation of : `Patches Are All You Need?
    <https://arxiv.org/pdf/2201.09792.pdf>`_

    Modified from the `official repo
    <https://github.com/locuslab/convmixer/blob/main/convmixer.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convmixer.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvMixer.arch_settings``. And if dict, it
            should include the following two keys:

            - embed_dims (int): The dimensions of patch embedding.
            - depth (int): Number of repetitions of ConvMixer Layer.
            - patch_size (int): The patch size.
            - kernel_size (int): The kernel size of depthwise conv layers.

            Defaults to '768/32'.
        in_channels (int): Number of input image channels. Defaults to 3.
        patch_size (int): The size of one patch in the patch embed layer.
            Defaults to 7.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): Initialization config dict.
    """
    arch_settings = {
        '768/32': {
            'embed_dims': 768,
            'depth': 32,
            'patch_size': 7,
            'kernel_size': 7
        },
        '1024/20': {
            'embed_dims': 1024,
            'depth': 20,
            'patch_size': 14,
            'kernel_size': 9
        },
        '1536/20': {
            'embed_dims': 1536,
            'depth': 20,
            'patch_size': 7,
            'kernel_size': 9
        },
    }

    def __init__(self,
                 arch='768/32',
                 in_channels=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 out_indices=-1,
                 frozen_stages=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            essential_keys = {
                'embed_dims', 'depth', 'patch_size', 'kernel_size'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'

        self.embed_dims = arch['embed_dims']
        self.depth = arch['depth']
        self.patch_size = arch['patch_size']
        self.kernel_size = arch['kernel_size']
        self.act = build_activation_layer(act_cfg)

        # check out indices and frozen stages
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.depth + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Set stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.embed_dims,
                kernel_size=self.patch_size,
                stride=self.patch_size), self.act,
            build_norm_layer(norm_cfg, self.embed_dims)[1])

        # Set conv2d according to torch version
        convfunc = nn.Conv2d
        if digit_version(torch.__version__) < digit_version('1.9.0'):
            convfunc = Conv2dAdaptivePadding

        # Repetitions of ConvMixer Layer
        self.stages = nn.Sequential(*[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        convfunc(
                            self.embed_dims,
                            self.embed_dims,
                            self.kernel_size,
                            groups=self.embed_dims,
                            padding='same'), self.act,
                        build_norm_layer(norm_cfg, self.embed_dims)[1])),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
                self.act,
                build_norm_layer(norm_cfg, self.embed_dims)[1])
            for _ in range(self.depth)
        ])

        self._freeze_stages()

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        # x = self.pooling(x).flatten(1)
        return tuple(outs)

    def train(self, mode=True):
        super(ConvMixer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            stage = self.stages[i]
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False
