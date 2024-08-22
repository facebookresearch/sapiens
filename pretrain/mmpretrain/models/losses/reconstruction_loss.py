# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class PixelReconstructionLoss(BaseModule):
    """Loss for the reconstruction of pixel in Masked Image Modeling.

    This module measures the distance between the target image and the
    reconstructed image and compute the loss to optimize the model. Currently,
    This module only provides L1 and L2 loss to penalize the reconstructed
    error. In addition, a mask can be passed in the ``forward`` function to
    only apply loss on visible region, like that in MAE.

    Args:
        criterion (str): The loss the penalize the reconstructed error.
            Currently, only supports L1 and L2 loss
        channel (int, optional): The number of channels to average the
            reconstruction loss. If not None, the reconstruction loss
            will be divided by the channel. Defaults to None.
    """

    def __init__(self, criterion: str, channel: Optional[int] = None) -> None:
        super().__init__()

        if criterion == 'L1':
            self.penalty = torch.nn.L1Loss(reduction='none')
        elif criterion == 'L2':
            self.penalty = torch.nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError(f'Currently, PixelReconstructionLoss \
            only supports L1 and L2 loss, but get {criterion}')

        self.channel = channel if channel is not None else 1

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function to compute the reconstrction loss.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        loss = self.penalty(pred, target)

        # if the dim of the loss is 3, take the average of the loss
        # along the last dim
        if len(loss.shape) == 3:
            loss = loss.mean(dim=-1)

        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum() / self.channel

        return loss
