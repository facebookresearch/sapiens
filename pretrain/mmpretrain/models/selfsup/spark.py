# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils.norm import build_norm_layer
from ..utils.sparse_modules import SparseHelper
from .base import BaseSelfSupervisor


@MODELS.register_module()
class SparK(BaseSelfSupervisor):
    """Implementation of SparK.

    Implementation of `Designing BERT for Convolutional Networks: Sparse and
    Hierarchical Masked Modeling <https://arxiv.org/abs/2301.03580>`_.

    Modified from
    https://github.com/keyu-tian/SparK/blob/main/pretrain/spark.py
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        head: dict,
        pretrained: Optional[str] = None,
        data_preprocessor: Optional[dict] = None,
        input_size: int = 224,
        downsample_raito: int = 32,
        mask_ratio: float = 0.6,
        enc_dec_norm_cfg=dict(type='SparseSyncBatchNorm2d'),
        enc_dec_norm_dim: int = 2048,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.input_size = input_size
        self.downsample_raito = downsample_raito
        feature_map_size = input_size // downsample_raito
        self.feature_map_size = feature_map_size

        self.mask_ratio = mask_ratio
        self.len_keep = round(feature_map_size * feature_map_size *
                              (1 - mask_ratio))

        self.enc_dec_norm_cfg = enc_dec_norm_cfg
        self.enc_dec_norms = nn.ModuleList()
        self.enc_dec_projectors = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        proj_out_dim = self.neck.feature_dim
        for i in range(len(self.backbone.out_indices)):
            enc_dec_norm = build_norm_layer(self.enc_dec_norm_cfg,
                                            enc_dec_norm_dim)
            self.enc_dec_norms.append(enc_dec_norm)

            kernel_size = 1 if i <= 0 else 3
            proj_layer = nn.Conv2d(
                enc_dec_norm_dim,
                proj_out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True)
            if i == 0 and enc_dec_norm_dim == proj_out_dim:
                proj_layer = nn.Identity()
            self.enc_dec_projectors.append(proj_layer)

            mask_token = nn.Parameter(torch.zeros(1, enc_dec_norm_dim, 1, 1))
            trunc_normal_(mask_token, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(mask_token)

            enc_dec_norm_dim //= 2
            proj_out_dim //= 2
            feature_map_size *= 2

    def mask(self,
             shape: torch.Size,
             device: Union[torch.device, str],
             generator: Optional[torch.Generator] = None):
        """Mask generation.

        Args:
            shape (torch.Size): The shape of the input images.
            device (Union[torch.device, str]): The device of the tensor.
            generator (torch.Generator, optional): Generator for random
                functions. Defaults to None
        Returns:
            torch.Tensor: The generated mask.
        """
        B, C, H, W = shape
        f = self.feature_map_size
        idx = torch.rand(B, f * f, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(
            B, f * f, dtype=torch.bool, device=device).scatter_(
                dim=1, index=idx, value=True).view(B, 1, f, f)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.
        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        # active mask of feature map, (B, 1, f, f)
        active_mask_feature_map = self.mask(inputs.shape, inputs.device)
        SparseHelper._cur_active = active_mask_feature_map

        # active mask of original input, (B, 1, H, W)
        active_mask_origin = active_mask_feature_map.repeat_interleave(
            self.downsample_raito,
            2).repeat_interleave(self.downsample_raito, 3)
        masked_img = inputs * active_mask_origin

        # get hierarchical encoded sparse features in a list
        # containing four feature maps
        feature_maps = self.backbone(masked_img)

        # from the smallest feature map to the largest
        feature_maps = list(feature_maps)
        feature_maps.reverse()

        cur_active = active_mask_feature_map
        feature_maps_to_dec = []
        for i, feature_map in enumerate(feature_maps):
            if feature_map is not None:
                # fill in empty positions with [mask] embeddings
                feature_map = self.enc_dec_norms[i](feature_map)
                mask_token = self.mask_tokens[i].expand_as(feature_map)
                feature_map = torch.where(
                    cur_active.expand_as(feature_map), feature_map,
                    mask_token.to(feature_map.dtype))
                feature_map = self.enc_dec_projectors[i](feature_map)
            feature_maps_to_dec.append(feature_map)

            # dilate the mask map
            cur_active = cur_active.repeat_interleave(
                2, dim=2).repeat_interleave(
                    2, dim=3)

        # decode and reconstruct
        rec_img = self.neck(feature_maps_to_dec)

        # compute loss
        loss = self.head(rec_img, inputs, active_mask_feature_map)
        losses = dict(loss=loss)
        return losses
