# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from torch.nn import functional as F

@MODELS.register_module()
class MAEPretrainHead2(BaseModule):
    """Head for MAE Pre-training.

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.loss_module = MODELS.build(loss)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        r"""Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images. The shape should
                be :math:`(B, 3, H, W)`.

        Returns:
            torch.Tensor: Patchified images. The shape is
            :math:`(B, L, \text{patch_size}^2 \times 3)`.
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""Combine non-overlapped patches into images.

        Args:
            x (torch.Tensor): The shape is
                :math:`(B, L, \text{patch_size}^2 \times 3)`.

        Returns:
            torch.Tensor: The shape is :math:`(B, 3, H, W)`.
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def construct_target(self, target: torch.Tensor, downsample_factor: int = 4) -> torch.Tensor: 
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.

        Args:
            target (torch.Tensor): Image with the shape of B x 3 x H x W

        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C
        """
        B, C, H, W = target.shape

        ## downsample the image from 4K to 1K
        target = F.interpolate(input=target, size=(H//downsample_factor, W//downsample_factor), mode='bicubic', align_corners=False)
        target = self.patchify(target) ## B x 3 x 1024 x 1024 -> B x 4096 x (16 x 16 x 3)

        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True) 
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        target = self.unpatchify(target) ## 1k spatial, B x 3 x H x W

        ## upsample the image from 1K to 4k
        target = F.interpolate(input=target, size=(H, W), mode='bicubic', align_corners=False) ## 1k -> 4k

        return target
    
    def get_norm_pix_mean_var(self, target):
        target = self.patchify(target)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        return mean, var

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = target.shape

        target = self.construct_target(target) # B x 3 x 4096 x 4096, normalized by mean and var (pixel intensities across channels. 64 patch size)
        pred = self.unpatchify(pred) ## B x 3 x 1024 x 1024

        if pred.shape[2] != target.shape[2]:
            pred = F.interpolate(input=pred, size=(H, W), mode='bicubic', align_corners=False) ## B x 3 x 4096 x 4096

        loss = self.loss_module(pred, target, mask=None)

        return loss
