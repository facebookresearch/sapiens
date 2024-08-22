# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.models.backbones.hivit import BlockWithRPE
from mmpretrain.registry import MODELS
from ..backbones.vision_transformer import TransformerEncoderLayer
from ..utils import build_2d_sincos_position_embedding


class PatchSplit(nn.Module):
    """The up-sample module used in neck (transformer pyramid network)

    Args:
        dim (int): the input dimension (channel number).
        fpn_dim (int): the fpn dimension (channel number).
        norm_cfg (dict): Config dict for normalization layer.
                Defaults to ``dict(type='LN')``.
    """

    def __init__(self, dim, fpn_dim, norm_cfg):
        super().__init__()
        _, self.norm = build_norm_layer(norm_cfg, dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(B, N, H, W, 2, 2,
                      self.fpn_dim).permute(0, 1, 2, 4, 3, 5,
                                            6).reshape(B, N, 2 * H, 2 * W,
                                                       self.fpn_dim)
        return x


@MODELS.register_module()
class iTPNPretrainDecoder(BaseModule):
    """The neck module of iTPN (transformer pyramid network).

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 512.
        fpn_dim (int): The fpn dimension (channel number).
        fpn_depth (int): The layer number of feature pyramid.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.
        reconstruction_type (str): The itpn supports 2 kinds of supervisions.
            Defaults to 'pixel'.
        num_outs (int): The output number of neck (transformer pyramid
            network). Defaults to 3.
        predict_feature_dim (int): The output dimension to supervision.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Initialization config
            dict. Defaults to None.
    """

    def __init__(self,
                 num_patches: int = 196,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 512,
                 fpn_dim: int = 256,
                 fpn_depth: int = 2,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 6,
                 decoder_num_heads: int = 16,
                 mlp_ratio: int = 4,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 reconstruction_type: str = 'pixel',
                 num_outs: int = 3,
                 qkv_bias: bool = True,
                 qk_scale: Optional[bool] = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 predict_feature_dim: Optional[float] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_patches = num_patches
        assert reconstruction_type in ['pixel', 'clip'], \
            'iTPN method only support `pixel` and `clip`, ' \
            f'but got `{reconstruction_type}`.'
        self.reconstruction_type = reconstruction_type
        self.num_outs = num_outs

        self.build_transformer_pyramid(
            num_outs=num_outs,
            embed_dim=embed_dim,
            fpn_dim=fpn_dim,
            fpn_depth=fpn_depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            rpe=False,
            norm_cfg=norm_cfg,
        )

        # merge the output
        self.decoder_embed = nn.ModuleList()
        self.decoder_embed.append(
            nn.Sequential(
                nn.LayerNorm(fpn_dim),
                nn.Linear(fpn_dim, decoder_embed_dim, bias=True),
            ))

        if self.num_outs >= 2:
            self.decoder_embed.append(
                nn.Sequential(
                    nn.LayerNorm(fpn_dim),
                    nn.Linear(fpn_dim, decoder_embed_dim // 4, bias=True),
                ))
        if self.num_outs >= 3:
            self.decoder_embed.append(
                nn.Sequential(
                    nn.LayerNorm(fpn_dim),
                    nn.Linear(fpn_dim, decoder_embed_dim // 16, bias=True),
                ))

        if reconstruction_type == 'pixel':
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))

            # create new position embedding, different from that in encoder
            # and is not learnable
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, decoder_embed_dim),
                requires_grad=False)

            self.decoder_blocks = nn.ModuleList([
                TransformerEncoderLayer(
                    decoder_embed_dim,
                    decoder_num_heads,
                    int(mlp_ratio * decoder_embed_dim),
                    qkv_bias=True,
                    norm_cfg=norm_cfg) for _ in range(decoder_depth)
            ])

            self.decoder_norm_name, decoder_norm = build_norm_layer(
                norm_cfg, decoder_embed_dim, postfix=1)
            self.add_module(self.decoder_norm_name, decoder_norm)

            # Used to map features to pixels
            if predict_feature_dim is None:
                predict_feature_dim = patch_size**2 * in_chans
            self.decoder_pred = nn.Linear(
                decoder_embed_dim, predict_feature_dim, bias=True)
        else:
            _, norm = build_norm_layer(norm_cfg, embed_dim)
            self.add_module('norm', norm)

    def build_transformer_pyramid(self,
                                  num_outs=3,
                                  embed_dim=512,
                                  fpn_dim=256,
                                  fpn_depth=2,
                                  mlp_ratio=4.0,
                                  qkv_bias=True,
                                  qk_scale=None,
                                  drop_rate=0.0,
                                  attn_drop_rate=0.0,
                                  rpe=False,
                                  norm_cfg=None):
        Hp = None
        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        if num_outs > 1:
            if embed_dim != fpn_dim:
                self.align_dim_16tofpn = nn.Linear(embed_dim, fpn_dim)
            else:
                self.align_dim_16tofpn = None
            self.fpn_modules = nn.ModuleList()
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg))
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=False,
                    norm_cfg=norm_cfg,
                ))

            self.align_dim_16to8 = nn.Linear(
                mlvl_dims['8'], fpn_dim, bias=False)
            self.split_16to8 = PatchSplit(mlvl_dims['16'], fpn_dim, norm_cfg)
            self.block_16to8 = nn.Sequential(*[
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg,
                ) for _ in range(fpn_depth)
            ])

        if num_outs > 2:
            self.align_dim_8to4 = nn.Linear(
                mlvl_dims['4'], fpn_dim, bias=False)
            self.split_8to4 = PatchSplit(fpn_dim, fpn_dim, norm_cfg)
            self.block_8to4 = nn.Sequential(*[
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg,
                ) for _ in range(fpn_depth)
            ])
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp,
                    fpn_dim,
                    0,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.,
                    rpe=rpe,
                    norm_cfg=norm_cfg))

    def init_weights(self) -> None:
        """Initialize position embedding and mask token of MAE decoder."""
        super().init_weights()

        if self.reconstruction_type == 'pixel':
            decoder_pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.decoder_pos_embed.shape[-1],
                cls_token=False)
            self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())

            torch.nn.init.normal_(self.mask_token, std=.02)
        else:
            self.rescale_init_weight()

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.fpn_modules):
            if isinstance(layer, BlockWithRPE):
                if layer.attn is not None:
                    rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def decoder_norm(self):
        """The normalization layer of decoder."""
        return getattr(self, self.decoder_norm_name)

    def forward(self,
                x: torch.Tensor,
                ids_restore: torch.Tensor = None) -> torch.Tensor:
        """The forward function.

        The process computes the visible patches' features vectors and the mask
        tokens to output feature vectors, which will be used for
        reconstruction.

        Args:
            x (torch.Tensor): hidden features, which is of shape
                    B x (L * mask_ratio) x C.
            ids_restore (torch.Tensor): ids to restore original image.

        Returns:
            torch.Tensor: The reconstructed feature vectors, which is of
            shape B x (num_patches) x C.
        """

        features = x[:2]
        x = x[-1]
        B, L, _ = x.shape
        x = x[..., None, None, :]
        Hp = Wp = math.sqrt(L)

        outs = [x] if self.align_dim_16tofpn is None else [
            self.align_dim_16tofpn(x)
        ]
        if self.num_outs >= 2:
            x = self.block_16to8(
                self.split_16to8(x) + self.align_dim_16to8(features[1]))
            outs.append(x)
        if self.num_outs >= 3:
            x = self.block_8to4(
                self.split_8to4(x) + self.align_dim_8to4(features[0]))
            outs.append(x)
        if self.num_outs > 3:
            outs = [
                out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(
                    0, 5, 1, 3, 2, 4).reshape(B, -1, Hp * out.shape[-3],
                                              Wp * out.shape[-2]).contiguous()
                for out in outs
            ]
            if self.num_outs >= 4:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))
            if self.num_outs >= 5:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))

        for i, out in enumerate(outs):
            out = self.fpn_modules[i](out)
            outs[i] = out

        if self.reconstruction_type == 'pixel':
            feats = []
            for feat, layer in zip(outs, self.decoder_embed):
                x = layer(feat).reshape(B, L, -1)
                # append mask tokens to sequence
                mask_tokens = self.mask_token.repeat(
                    x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
                x = torch.cat([x, mask_tokens], dim=1)
                x = torch.gather(
                    x,
                    dim=1,
                    index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                feats.append(x)
            x = feats.pop(0)
            # add pos embed
            x = x + self.decoder_pos_embed

            for i, feat in enumerate(feats):
                x = x + feats[i]
            # apply Transformer blocks
            for i, blk in enumerate(self.decoder_blocks):
                x = blk(x)
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            return x
        else:
            feats = []
            for feat, layer in zip(outs, self.decoder_embed):
                x = layer(feat).reshape(B, L, -1)
                feats.append(x)
            x = feats.pop(0)
            for i, feat in enumerate(feats):
                x = x + feats[i]

            x = self.norm(x)

            return x
