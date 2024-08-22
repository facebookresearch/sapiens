# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.models import VisionTransformer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor


@MODELS.register_module()
class HOGGenerator(BaseModule):
    """Generate HOG feature for images.

    This module is used in MaskFeat to generate HOG feature. The code is
    modified from file `slowfast/models/operators.py
    <https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/operators.py>`_.
    Here is the link of `HOG wikipedia
    <https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients>`_.

    Args:
        nbins (int): Number of bin. Defaults to 9.
        pool (float): Number of cell. Defaults to 8.
        gaussian_window (int): Size of gaussian kernel. Defaults to 16.
    """

    def __init__(self,
                 nbins: int = 9,
                 pool: int = 8,
                 gaussian_window: int = 16) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).contiguous()
        weight_y = weight_x.transpose(2, 3).contiguous()
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gaussian_kernel = self.get_gaussian_kernel(gaussian_window,
                                                       gaussian_window // 2)
            self.register_buffer('gaussian_kernel', gaussian_kernel)

    def get_gaussian_kernel(self, kernlen: int, std: int) -> torch.Tensor:
        """Returns a 2D Gaussian kernel array."""

        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        kernel_1d = _gaussian_fn(kernlen, std)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d / kernel_2d.sum()

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        """Reshape HOG Features for output."""
        hog_feat = hog_feat.flatten(1, 2)
        self.unfold_size = hog_feat.shape[-1] // 14
        hog_feat = hog_feat.permute(0, 2, 3, 1)
        hog_feat = hog_feat.unfold(1, self.unfold_size,
                                   self.unfold_size).unfold(
                                       2, self.unfold_size, self.unfold_size)
        hog_feat = hog_feat.flatten(1, 2).flatten(2)
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate hog feature for each batch images.

        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Hog features.
        """
        # input is RGB image with shape [B 3 H W]
        self.h, self.w = x.size(-2), x.size(-1)
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, 'h {} gw {}'.format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gaussian_kernel = self.gaussian_kernel.repeat(
                    [repeat_rate, repeat_rate])
            else:
                temp_gaussian_kernel = self.gaussian_kernel
            norm_rgb *= temp_gaussian_kernel

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        self.out = F.normalize(out, p=2, dim=2)

        return self._reshape(self.out)

    def generate_hog_image(self, hog_out: torch.Tensor) -> np.ndarray:
        """Generate HOG image according to HOG features."""
        assert hog_out.size(0) == 1 and hog_out.size(1) == 3, \
            'Check the input batch size and the channcel number, only support'\
            '"batch_size = 1".'
        hog_image = np.zeros([self.h, self.w])
        cell_gradient = np.array(hog_out.mean(dim=1).squeeze().detach().cpu())
        cell_width = self.pool / 2
        max_mag = np.array(cell_gradient).max()
        angle_gap = 360 / self.nbins

        for x in range(cell_gradient.shape[1]):
            for y in range(cell_gradient.shape[2]):
                cell_grad = cell_gradient[:, x, y]
                cell_grad /= max_mag
                angle = 0
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.pool +
                             magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.pool +
                             magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.pool -
                             magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.pool -
                             magnitude * cell_width * math.sin(angle_radian))
                    magnitude = 0 if magnitude < 0 else magnitude
                    cv2.line(hog_image, (y1, x1), (y2, x2),
                             int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return hog_image


@MODELS.register_module()
class MaskFeatViT(VisionTransformer):
    """Vision Transformer for MaskFeat pre-training.

    A PyTorch implement of: `Masked Feature Prediction for Self-Supervised
    Visual Pre-Training <https://arxiv.org/abs/2112.09133>`_.

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            It only works without input mask. Defaults to ``"avg_featmap"``.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'raw',
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            with_cls_token=True,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_token = nn.parameter.Parameter(
            torch.zeros(1, 1, self.embed_dims), requires_grad=True)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self) -> None:
        """Initialize position embedding, mask token and cls token."""
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):

            nn.init.trunc_normal_(self.cls_token, std=.02)
            nn.init.trunc_normal_(self.mask_token, std=.02)
            nn.init.trunc_normal_(self.pos_embed, std=.02)

            self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor, optional): Input masks.

        Returns:
            torch.Tensor: Features with cls_tokens.
        """
        if mask is None:
            return super().forward(x)

        else:
            B = x.shape[0]
            x = self.patch_embed(x)[0]

            # masking: length -> length * mask_ratio
            B, L, _ = x.shape
            mask_tokens = self.mask_token.expand(B, L, -1)
            mask = mask.unsqueeze(-1)
            x = x * (1 - mask.int()) + mask_tokens * mask

            # append cls token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.drop_after_pos(x)

            for i, layer in enumerate(self.layers):
                x = layer(x)

                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.norm1(x)

            return x


@MODELS.register_module()
class MaskFeat(BaseSelfSupervisor):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2112.09133>`_.
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
        mask = torch.stack([data_sample.mask for data_sample in data_samples])
        mask = mask.flatten(1).bool()

        latent = self.backbone(inputs, mask)
        B, L, C = latent.shape
        pred = self.neck((latent.view(B * L, C), ))
        pred = pred[0].view(B, L, -1)
        hog = self.target_generator(inputs)

        # remove cls_token before compute loss
        loss = self.head.loss(pred[:, 1:], hog, mask)
        losses = dict(loss=loss)
        return losses
