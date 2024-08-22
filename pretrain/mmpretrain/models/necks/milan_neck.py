# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from torch import nn

from mmpretrain.registry import MODELS
from ..backbones.vision_transformer import TransformerEncoderLayer
from ..utils import PromptMultiheadAttention
from .mae_neck import MAEPretrainDecoder


class PromptTransformerEncoderLayer(TransformerEncoderLayer):
    """Prompt Transformer Encoder Layer for MILAN.

    This module is specific for the prompt encoder in MILAN. It will not update
    the visible tokens from the encoder.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): Enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels=int,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.attn = PromptMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

    def forward(self, x: torch.Tensor, visible_tokens: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """Forward function for `PromptMultiheadAttention`.

        Args:
            x (torch.Tensor): Mask token features with shape N x L_m x C.
            visible_tokens (torch.Tensor): The visible tokens features from
                encoder with shape N x L_v x C.
            ids_restore (torch.Tensor): The ids of all tokens in the original
                image with shape N x L.

        Returns:
            torch Tensor: Output features with shape N x L x C.
        """
        x = x + self.attn(self.norm1(x), visible_tokens, ids_restore)
        x = self.ffn(self.norm2(x), identity=x)
        return x


@MODELS.register_module()
class MILANPretrainDecoder(MAEPretrainDecoder):
    """Prompt decoder for MILAN.

    This decoder is used in MILAN pretraining, which will not update these
    visible tokens from the encoder.

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        predict_feature_dim (int): The dimension of the feature to be
            predicted. Defaults to 512.
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
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 predict_feature_dim: int = 512,
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

        # map the dim of features from decoder to the dim compatible with
        # that of CLIP
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, predict_feature_dim, bias=True)

        # use prompt transformer encoder layer, instead of the conventional
        # transformer encoder layer
        self.decoder_blocks = nn.ModuleList([
            PromptTransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor,
                ids_keep: torch.Tensor,
                ids_dump: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): The input features, which is of shape (N, L, C).
            ids_restore (torch.Tensor): The indices to restore these tokens
                to the original image.
            ids_keep (torch.Tensor): The indices of tokens to be kept.
            ids_dump (torch.Tensor): The indices of tokens to be masked.

        Returns:
            torch.Tensor: The reconstructed features, which is of shape
            (N, L, C).
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # split mask tokens and visible tokens
        visible_tokens = torch.cat([
            x[:, :1, :],
            torch.gather(
                x[:, 1:, :],
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        ],
                                   dim=1)
        x = torch.gather(
            x[:, 1:, :],
            dim=1,
            index=ids_dump.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        for blk in self.decoder_blocks:
            x = blk(x, visible_tokens, ids_restore)

        # full sequence recovery
        x_ = torch.cat([visible_tokens[:, 1:, :], x], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                   x.shape[-1]))  # unshuffle
        x = torch.cat([visible_tokens[:, :1, :], x_], dim=1)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x
