# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class SparKPretrainHead(BaseModule):
    """Pre-training head for SparK.

    Args:
        loss (dict): Config of loss.
        norm_pix (bool): Whether or not normalize target. Defaults to True.
        patch_size (int): Patch size, equal to downsample ratio of backbone.
            Defaults to 32.
    """

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = True,
                 patch_size: int = 32) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.loss = MODELS.build(loss)

    def patchify(self, imgs):
        """Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images, of shape B x C x H x W.
        Returns:
            torch.Tensor: Patchified images. The shape is B x L x D.
        """
        p = self.patch_size
        assert len(imgs.shape
                   ) == 4 and imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        B, C, ori_h, ori_w = imgs.shape
        h = ori_h // p
        w = ori_w // p
        x = imgs.reshape(shape=(B, C, h, p, w, p))
        x = torch.einsum('bchpwq->bhwpqc', x)

        # (B, f*f, downsample_raito*downsample_raito*3)
        x = x.reshape(shape=(B, h * w, p**2 * C))
        return x

    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.
        Args:
            target (torch.Tensor): Image with the shape of B x 3 x H x W
        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C
        """
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                active_mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MAE head.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            active_mask (torch.Tensor): The mask of the target image.
        Returns:
            torch.Tensor: The reconstruction loss.
        """
        # (B, C, H, W) -> (B, L, C) and perform normalization
        target = self.construct_target(target)

        # (B, C, H, W) -> (B, L, C)
        pred = self.patchify(pred)

        # (B, 1, f, f) -> (B, L)
        non_active_mask = active_mask.logical_not().int().view(
            active_mask.shape[0], -1)

        # MSE loss on masked patches
        loss = self.loss(pred, target, non_active_mask)
        return loss
