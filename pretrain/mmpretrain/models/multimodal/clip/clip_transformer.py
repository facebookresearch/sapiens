# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/zejiangh/MILAN
from typing import Optional, Tuple

import torch
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.models.utils.clip_generator_helper import \
    ResidualAttentionBlock
from mmpretrain.registry import MODELS


@MODELS.register_module()
class CLIPTransformer(nn.Module):
    """Transformer.

    Both visual and text branches use this transformer.

    Args:
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
    """

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        for _ in range(layers - 1):
            self.resblocks.append(
                ResidualAttentionBlock(width, heads, attn_mask))
        self.resblocks.append(
            ResidualAttentionBlock(
                width, heads, attn_mask, return_attention=True))

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function."""
        z = []
        for idx, blk in enumerate(self.resblocks):
            if idx < self.layers - 1:
                x = blk(x)
                z.append(x.permute(1, 0, 2))
            else:
                x, attention = blk(x)
                z.append(x.permute(1, 0, 2))
        return x, attention, z


@MODELS.register_module()
class CLIPProjection(BaseModule):
    """Neck with CLIP Projection.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_cfg: Optional[dict] = None):
        super(CLIPProjection, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = in_channels**-0.5
        self.proj = nn.Parameter(scale *
                                 torch.randn(in_channels, out_channels))

    def forward(self, inputs: Tuple) -> Tuple[torch.Tensor]:
        """forward function.

        Args:
            inputs (Tuple): The features extracted from
                 the backbone. Multiple stage inputs are acceptable but only
                  the last stage will be used.
        Returns:
            Tuple(torch.Tensor)): A tuple of reducted features.
        """
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
            out = inputs @ self.proj
        elif isinstance(inputs, torch.Tensor):
            out = inputs @ self.proj
        else:
            raise TypeError(
                '`CLIPProjection` neck inputs should be tuple or torch.tensor')
        return (out, )
