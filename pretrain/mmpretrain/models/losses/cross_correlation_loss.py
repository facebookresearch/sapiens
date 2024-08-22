# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CrossCorrelationLoss(BaseModule):
    """Cross correlation loss function.

    Compute the on-diagnal and off-diagnal loss.

    Args:
        lambd (float): The weight for the off-diag loss.
    """

    def __init__(self, lambd: float = 0.0051) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, cross_correlation_matrix: torch.Tensor) -> torch.Tensor:
        """Forward function of cross correlation loss.

        Args:
            cross_correlation_matrix (torch.Tensor): The cross correlation
                matrix.

        Returns:
            torch.Tensor: cross correlation loss.
        """
        # loss
        on_diag = torch.diagonal(cross_correlation_matrix).add_(-1).pow_(
            2).sum()
        off_diag = self.off_diagonal(cross_correlation_matrix).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """Rreturn a flattened view of the off-diagonal elements of a square
        matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
