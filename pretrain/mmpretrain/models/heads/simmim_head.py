# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class SimMIMHead(BaseModule):
    """Head for SimMIM Pre-training.

    Args:
        patch_size (int): Patch size of each token.
        loss (dict): The config for loss.
    """

    def __init__(self, patch_size: int, loss: dict) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.loss_module = MODELS.build(loss)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        This method will expand mask to the size of the original image.

        Args:
            pred (torch.Tensor): The reconstructed image (B, C, H, W).
            target (torch.Tensor): The target image (B, C, H, W).
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            self.patch_size, 2).unsqueeze(1).contiguous()
        loss = self.loss_module(pred, target, mask)

        return loss
