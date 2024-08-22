# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from .utils import weight_reduce_loss
import torch.nn.functional as F

@MODELS.register_module()
class EdgeAwareLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, eps=-100, thres_eps=1e-6, loss_name='loss_edge_aware'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.thres_eps = thres_eps
        self._loss_name = loss_name

    def forward(self, pred, target, valid_mask=None, weight=None, avg_factor=None, reduction_override=None):
        assert pred.shape == target.shape, f'The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        reduction = reduction_override if reduction_override else self.reduction
        if valid_mask is None:
            valid_mask = (target > self.eps).detach().float()
            valid_mask = valid_mask[:, 0, :, :].unsqueeze(1) ## B x 1 x H x W

        if valid_mask.sum() == 0:
            return 0.0 * pred.sum()

        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Adjust valid mask for gradients
        valid_mask_dx = valid_mask[:, :, :, :-1] * valid_mask[:, :, :, 1:]
        valid_mask_dy = valid_mask[:, :, :-1, :] * valid_mask[:, :, 1:, :]

        # Compute edge-aware loss
        loss_dx = F.mse_loss(pred_dx, target_dx, reduction='none')
        loss_dy = F.mse_loss(pred_dy, target_dy, reduction='none')

        # Apply valid mask before reduction
        loss_dx = (loss_dx * valid_mask_dx).sum() / valid_mask_dx.sum().clamp(min=1)
        loss_dy = (loss_dy * valid_mask_dy).sum() / valid_mask_dy.sum().clamp(min=1)

        # Combine losses
        loss = loss_dx + loss_dy

        if reduction == 'mean':
            pass  # Loss is already averaged
        elif reduction == 'sum':
            loss = loss * pred.shape[0]  # Multiply by batch size to get sum
        elif reduction == 'none':
            loss = loss.expand(pred.shape[0], -1, -1, -1)  # Expand loss to match input shape
        else:
            raise ValueError(f'Invalid reduction type: {reduction}')

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight

        # Convert nan to 0
        loss = torch.nan_to_num(loss, 
                nan=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                posinf=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                neginf=torch.tensor(0, dtype=pred.dtype, device=pred.device))
        
        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name
