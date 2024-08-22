# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from mmpretrain.registry import MODELS
from .mae_head import MAEPretrainHead


@MODELS.register_module()
class MixMIMPretrainHead(MAEPretrainHead):
    """Head for MixMIM Pre-training.

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
        super().__init__(loss=loss, norm_pix=norm_pix, patch_size=patch_size)

    def loss(self, x_rec: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        target = self.construct_target(target)

        B, L, C = x_rec.shape

        # unmix tokens
        x1_rec = x_rec[:B // 2]
        x2_rec = x_rec[B // 2:]

        unmix_x_rec = x1_rec * mask + x2_rec.flip(0) * (1 - mask)

        loss_rec = self.loss_module(unmix_x_rec, target)

        return loss_rec
