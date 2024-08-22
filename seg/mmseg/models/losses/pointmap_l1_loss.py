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
class PointmapL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, eps=-100, normalize=False, loss_name='loss_l1'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name
        self.normalize = normalize

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert pred.shape == target.shape, f'The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        B, C, H, W = target.shape

        reduction = reduction_override if reduction_override else self.reduction
        valid_mask = (target > self.eps).detach().float()
        valid_mask = valid_mask[:, 0, :, :].unsqueeze(1) ## B x 1 x H x W

        if self.normalize:
            pred_norm = torch.linalg.vector_norm(pred, dim=1, keepdim=True) ## B x 1 x H x W
            avg_pred_norm = (pred_norm * valid_mask).view(B, -1).sum(dim=1) / valid_mask.view(B, -1).sum(dim=1).clamp(min=1) ## B x 1
            target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True) ## B x 1 x H x W
            avg_target_norm = (target_norm * valid_mask).view(B, -1).sum(dim=1) / valid_mask.view(B, -1).sum(dim=1).clamp(min=1) ## B x 1

            pred = pred / (avg_pred_norm + 1e-8)
            target = target / (avg_target_norm + 1e-8)

        loss = F.l1_loss(pred, target, reduction='none') * valid_mask

        if reduction == 'mean':
            loss = loss.sum() / valid_mask.sum().clamp(min=1)
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            pass  # Keep per-pixel loss
        else:
            raise ValueError(f'Invalid reduction type: {reduction}')

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name
