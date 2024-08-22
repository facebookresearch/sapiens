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
class StereoPointmapCorrespondenceLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='corresp_loss'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred1, pred2, batch_data_samples1, weight=None, avg_factor=None, reduction_override=None):
        assert pred1.shape == pred2.shape, f'The shapes of pred1 ({pred1.shape}) and pred2 ({pred2.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        B, C, H, W = pred1.shape

        device = pred1.device

        loss = 0
        num_correspondences = 0

        for i in range(B):
            pred_pointmap1 = pred1[i] ## 3 x H x W
            pred_pointmap2 = pred2[i] ## 3 x H x W

            ## compute correspondence loss. l2 distance between pointsmaps of correpoending points in pointmap is zero.
            pixel_coords1 = torch.from_numpy(batch_data_samples1[i].pixel_coords1).to(device)
            pixel_coords2 = torch.from_numpy(batch_data_samples1[i].pixel_coords2).to(device)

            assert pixel_coords1.shape[0] == pixel_coords2.shape[0]

            pred_points1 = pred_pointmap1[:, pixel_coords1[:, 1], pixel_coords1[:, 0]].T ## num_pixels x 3
            pred_points2 = pred_pointmap2[:, pixel_coords2[:, 1], pixel_coords2[:, 0]].T ## num_pixels x 3

            if len(pred_points1) == 0:
                continue

            num_correspondences += len(pred_points1)

            this_loss = F.mse_loss(pred_points1, pred_points2)
            loss += this_loss

        reduction = reduction_override if reduction_override else self.reduction

        if num_correspondences == 0:
            return torch.tensor(loss, device=device) * self.loss_weight

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name
