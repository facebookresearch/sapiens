# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict,
                 temperature: float = 0.1,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss_module = MODELS.build(loss)
        self.temperature = temperature

    def loss(self, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Forward function to compute contrastive loss.

        Args:
            pos (torch.Tensor): Nx1 positive similarity.
            neg (torch.Tensor): Nxk negative similarity.

        Returns:
            torch.Tensor: The contrastive loss.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)

        loss = self.loss_module(logits, labels)
        return loss
