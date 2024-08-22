# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.models.backbones.hivit import BlockWithRPE, HiViT, PatchMerge
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor


@MODELS.register_module()
class iTPNHiViT(HiViT):
    """HiViT for iTPN pre-training.

    Args:
        img_size (int | tuple): Input image size. Defaults to 224.
        patch_size (int | tuple): The patch size. Defaults to 16.
        inner_patches (int): Inner patch. Defaults to 4.
        stem_mlp_ratio (int): Ratio of MLP hidden dim to embedding dim
            in the first two stages. Defaults to 3.
        mlp_ratio (int): Ratio of MLP hidden dim to embedding dim in
            the last stage. Defaults to 4.
        qkv_bias (bool): Enable bias for qkv projections if True.
        qk_scale (float): The number of divider after q@k. Default to None.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        ape (bool): If True, add absolute position embedding to
            the patch embedding.
        rpe (bool): If True, add relative position embedding to
            the patch embedding.
        layer_scale_init_value (float): Layer-scale init values. Defaults to 0.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
        reconstruction_type (str): The reconstruction of self-supervised
            learning. Defaults to 'pixel'.
    """

    def __init__(
        self,
        arch='base',
        img_size: int = 224,
        patch_size: int = 16,
        inner_patches: int = 4,
        stem_mlp_ratio: int = 3.,
        mlp_ratio: int = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        ape: bool = True,
        rpe: bool = False,
        layer_scale_init_value: float = 0.0,
        mask_ratio: float = 0.75,
        reconstruction_type: str = 'pixel',
    ):
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            stem_mlp_ratio=stem_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            ape=ape,
            rpe=rpe,
            layer_scale_init_value=layer_scale_init_value)

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio

        assert reconstruction_type in ['pixel', 'clip'], \
            'iTPN method only support `pixel` and `clip`, ' \
            f'but got `{reconstruction_type}`.'
        self.reconstruction_type = reconstruction_type
        self.num_patches = self.patch_embed.num_patches

        if reconstruction_type == 'clip':
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().apply(self._init_weights)

        if self.reconstruction_type == 'clip':
            trunc_normal_(self.mask_token, std=0.02)
            self.rescale_init_weight()
        else:
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=False)
            self.pos_embed.data.copy_(pos_embed.float())

            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if isinstance(layer, BlockWithRPE):
                if layer.attn is not None:
                    rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(
            N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward_pixel(
        self,
        x: torch.Tensor,
        mask: Optional[bool] = True
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B, C, H, W = x.shape
            ids_keep, ids_restore, mask = self.masking_id(B, self.mask_ratio)

            x = self.patch_embed(x)

            x = torch.gather(
                x,
                dim=1,
                index=ids_keep[:, :, None, None,
                               None].expand(-1, -1, *x.shape[2:]))

            outs = []
            for blk in self.blocks[:-self.num_main_blocks]:
                if isinstance(blk, PatchMerge):
                    outs.append(x)
                x = blk(x)

            x = x[..., 0, 0, :]
            if self.ape:
                pos_embed = self.interpolate_pos_encoding(x, H, W)
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1,
                                                      pos_embed.shape[2]),
                )
                x = x + pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x)

            outs.append(x)

            return (tuple(outs), mask, ids_restore)

    def forward_clip(self,
                     x: torch.Tensor,
                     mask: Optional[bool] = True) -> Tuple:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B, C, H, W = x.shape
            x = self.patch_embed(x)

            outs = []
            for blk in self.blocks[:-self.num_main_blocks]:
                if isinstance(blk, PatchMerge):
                    outs.append(x)
                x = blk(x)

            x = x[..., 0, 0, :]
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            if self.ape:
                pos_embed = self.interpolate_pos_encoding(x, H, W)
                x = x + pos_embed
            x = self.pos_drop(x)

            rpe_index = True if self.rpe else None

            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x, rpe_index)

            outs.append(x)

            return tuple(outs)

    def forward(self, x: torch.Tensor, mask: Optional[bool] = True) -> Tuple:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """

        if self.reconstruction_type == 'pixel':
            return self.forward_pixel(x, mask)
        return self.forward_clip(x, mask)


@MODELS.register_module()
class iTPN(BaseSelfSupervisor):
    """iTPN.

    Implementation of `iTPN: Integrally Pre-Trained Transformer Pyramid
    Networks <https://arxiv.org/abs/2211.12735>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        if self.backbone.reconstruction_type == 'pixel':
            latent, mask, ids_restore = self.backbone(inputs)
            pred = self.neck(latent, ids_restore)

            loss = self.head.loss(pred, inputs, mask)
        else:
            mask = torch.stack(
                [data_sample.mask for data_sample in data_samples])

            img_latent = self.backbone(inputs[0], mask)

            # inputs[1] is the target image
            with torch.no_grad():
                target = self.target_generator(inputs[1])[0]
                target = target.detach()

            # iTPN contains a neck module
            feats = self.neck(img_latent)
            loss = self.head.loss(feats, target[:, 1:, :], mask)

        losses = dict(loss=loss)
        return losses
