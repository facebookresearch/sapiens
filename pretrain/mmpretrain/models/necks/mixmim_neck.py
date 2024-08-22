# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from ..utils import build_2d_sincos_position_embedding
from .mae_neck import MAEPretrainDecoder


@MODELS.register_module()
class MixMIMPretrainDecoder(MAEPretrainDecoder):
    """Decoder for MixMIM Pretraining.

    Some of the code is borrowed from `https://github.com/Sense-X/MixMIM`. # noqa

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        encoder_stride (int): The output stride of MixMIM backbone. Defaults
            to 32.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 num_patches: int = 196,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 encoder_stride: int = 32,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:

        super().__init__(
            num_patches=num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim),
            requires_grad=False)
        self.decoder_pred = nn.Linear(decoder_embed_dim, encoder_stride**2 * 3)

    def init_weights(self) -> None:
        """Initialize position embedding and mask token of MixMIM decoder."""
        super(MAEPretrainDecoder, self).init_weights()

        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.decoder_pos_embed.shape[-1],
            cls_token=False)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): The input features, which is of shape (N, L, C).
            mask (torch.Tensor): The tensor to indicate which tokens a
                re masked.

        Returns:
            torch.Tensor: The reconstructed features, which is of shape
            (N, L, C).
        """

        x = self.decoder_embed(x)
        B, L, C = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        x1 = x * (1 - mask) + mask_tokens * mask
        x2 = x * mask + mask_tokens * (1 - mask)
        x = torch.cat([x1, x2], dim=0)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for idx, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x
