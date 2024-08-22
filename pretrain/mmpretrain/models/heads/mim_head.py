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
class MIMHead(BaseModule):
    """Pre-training head for Masked Image Modeling.

    Args:
        loss (dict): Config dict for module of loss functions.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)

    def loss(self,
             pred: torch.Tensor,
             target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward head.

        Args:
            pred (torch.Tensor): Predictions with shape B x L x C.
            target (torch.Tensor): Targets with shape B x L x C.
            mask (torch.Tensor): Mask with shape B x L.

        Returns:
            torch.Tensor: The loss tensor.
        """
        loss = self.loss_module(pred, target, mask)
        return loss
