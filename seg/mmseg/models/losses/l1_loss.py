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
class MetricDepthL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, eps=-100, loss_name='loss_l1'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert pred.shape == target.shape, f'The shapes of pred ({pred.shape}) and target ({target.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        ## pred is B x H x W
        ## target is B x H x W

        reduction = reduction_override if reduction_override else self.reduction
        valid_mask = (target > self.eps).detach().float()
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

@MODELS.register_module()
class L1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, eps=-100, loss_name='loss_l1'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
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

        ## convert nan to 0
        loss = torch.nan_to_num(loss, 
                nan=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                posinf=torch.tensor(0, dtype=pred.dtype, device=pred.device), 
                neginf=torch.tensor(0, dtype=pred.dtype, device=pred.device))

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name
