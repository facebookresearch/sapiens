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
import cv2
import numpy as np


@MODELS.register_module()
class PointmapConsistencyLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=-100,
                 loss_name='loss_consistency'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        pred,
        target,
        gt_K=None,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert pred.shape == target.shape, 'the shapes of pred ' \
            f'({pred.shape}) and target ({target.shape}) are mismatch'

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        pred_X = pred[:, 0, :, :] ## B x H x W
        pred_Y = pred[:, 1, :, :] ## B x H x W
        pred_Z = pred[:, 2, :, :] ## B x H x W
        target_Z = target[:, 2, :, :]

        valid_mask = (target_Z > self.eps).detach().float() ## B x H x W

        # Get dimensions
        B, H, W = target_Z.shape
        device = target_Z.device
        cols = torch.arange(W, device=device).repeat(B, H, 1)  # B x H x W
        rows = torch.arange(H, device=device).repeat(B, W, 1).transpose(1, 2)  # B x H x W

        # Compute x and y from z and K
        x = (cols - gt_K[:, 0, 2].view(B, 1, 1)) * pred_Z / gt_K[:, 0, 0].view(B, 1, 1)
        y = (rows - gt_K[:, 1, 2].view(B, 1, 1)) * pred_Z / gt_K[:, 1, 1].view(B, 1, 1)

        ##---------------to debug consistency-------------------
        # target_X = target[:, 0, :, :]
        # target_Y = target[:, 1, :, :]
        # x = (cols - gt_K[:, 0, 2].view(B, 1, 1)) * target_Z / gt_K[:, 0, 0].view(B, 1, 1)
        # y = (rows - gt_K[:, 1, 2].view(B, 1, 1)) * target_Z / gt_K[:, 1, 1].view(B, 1, 1)

        # loss_X = torch.abs(target_X - x) * valid_mask
        # loss_Y = torch.abs(target_Y - y) * valid_mask

        ##-----------------------------------------------------
        # Loss calculations
        loss_X = torch.abs(pred_X - x) * valid_mask
        loss_Y = torch.abs(pred_Y - y) * valid_mask

        # Apply reduction
        if self.reduction == 'mean':
            loss = (loss_X + loss_Y).sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == 'sum':
            loss = (loss_X + loss_Y).sum()
        elif self.reduction == 'none':
            loss = loss_X + loss_Y
        else:
            raise ValueError('Unsupported reduction type')

        # Handle NaN values
        loss = torch.nan_to_num(loss, nan=0.0)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
