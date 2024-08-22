# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.models.backbones.beit import BEiTTransformerEncoderLayer
from mmpretrain.registry import MODELS


@MODELS.register_module()
class BEiTV2Neck(BaseModule):
    """Neck for BEiTV2 Pre-training.

    This module construct the decoder for the final prediction.

    Args:
        num_layers (int): Number of encoder layers of neck. Defaults to 2.
        early_layers (int): The layer index of the early output from the
            backbone. Defaults to 9.
        backbone_arch (str): Vision Transformer architecture. Defaults to base.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): The initialization value for the
            learnable scaling of attention and FFN. Defaults to 0.1.
        use_rel_pos_bias (bool): Whether to use unique relative position bias,
            if False, use shared relative position bias defined in backbone.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'depth': 12,
                'num_heads': 12,
                'feedforward_channels': 3072,
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'depth': 24,
                'num_heads': 16,
                'feedforward_channels': 4096,
            }),
    }

    def __init__(
        self,
        num_layers: int = 2,
        early_layers: int = 9,
        backbone_arch: str = 'base',
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 0.1,
        use_rel_pos_bias: bool = False,
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        init_cfg: Optional[Union[dict, List[dict]]] = dict(
            type='TruncNormal', layer='Linear', std=0.02, bias=0)
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if isinstance(backbone_arch, str):
            backbone_arch = backbone_arch.lower()
            assert backbone_arch in set(self.arch_zoo), \
                (f'Arch {backbone_arch} is not in default archs '
                 f'{set(self.arch_zoo)}')
            self.arch_settings = self.arch_zoo[backbone_arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(backbone_arch, dict) and essential_keys <= set(
                backbone_arch
            ), f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = backbone_arch

        # stochastic depth decay rule
        self.early_layers = early_layers
        depth = self.arch_settings['depth']
        dpr = np.linspace(0, drop_path_rate,
                          max(depth, early_layers + num_layers))

        self.patch_aggregation = nn.ModuleList()
        for i in range(early_layers, early_layers + num_layers):
            _layer_cfg = dict(
                embed_dims=self.arch_settings['embed_dims'],
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                layer_scale_init_value=layer_scale_init_value,
                window_size=None,
                use_rel_pos_bias=use_rel_pos_bias)
            self.patch_aggregation.append(
                BEiTTransformerEncoderLayer(**_layer_cfg))

        self.rescale_patch_aggregation_init_weight()

        embed_dims = self.arch_settings['embed_dims']
        _, norm = build_norm_layer(norm_cfg, embed_dims)
        self.add_module('norm', norm)

    def rescale_patch_aggregation_init_weight(self):
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.patch_aggregation):
            rescale(layer.attn.proj.weight.data,
                    self.early_layers + layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data,
                    self.early_layers + layer_id + 1)

    def forward(self, inputs: Tuple[torch.Tensor], rel_pos_bias: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the latent prediction and final prediction.

        Args:
            x (Tuple[torch.Tensor]): Features of tokens.
            rel_pos_bias (torch.Tensor): Shared relative position bias table.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - ``x``: The final layer features from backbone, which are normed
                in ``BEiTV2Neck``.
              - ``x_cls_pt``: The early state features from backbone, which are
                consist of final layer cls_token and early state patch_tokens
                from backbone and sent to PatchAggregation layers in the neck.
        """

        early_states, x = inputs[0], inputs[1]
        x_cls_pt = torch.cat([x[:, [0]], early_states[:, 1:]], dim=1)
        for layer in self.patch_aggregation:
            x_cls_pt = layer(x_cls_pt, rel_pos_bias=rel_pos_bias)

        # shared norm
        x, x_cls_pt = self.norm(x), self.norm(x_cls_pt)

        # remove cls_token
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        return x, x_cls_pt
