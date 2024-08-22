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
class StereoCorrespondencesLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, temperature=0.07, eps=1e-8, max_pixels=40000, loss_name='info_nce_loss'):
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
        self.eps = eps
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.max_pixels = int(max_pixels)  
    
    def get_similarities(self, desc1, desc2, euc=False):
        if euc:  # euclidean distance in same range than similarities
            dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
            sim = 1 / (1 + dists)
        else:
            # Compute similarities
            sim = desc1 @ desc2.transpose(-2, -1)
        return sim

    def forward(self, batch_desc1, batch_desc2, batch_data_samples1, weight=None, avg_factor=None, reduction_override=None):
        assert batch_desc1.shape == batch_desc2.shape, f'The shapes of batch_desc1 ({batch_desc1.shape}) and batch_desc2 ({batch_desc2.shape}) are mismatched'
        assert reduction_override in (None, 'none', 'mean', 'sum'), 'Invalid reduction_override value'

        B, C, H, W = batch_desc1.shape
        device = batch_desc1.device

        loss = 0
        num_correspondences = 0

        for i in range(B):
            desc1 = batch_desc1[i] ## 32 x H x W
            desc2 = batch_desc2[i] ## 32 x H x W

            ## compute infoNCE loss. 
            pixel_coords1 = torch.from_numpy(batch_data_samples1[i].pixel_coords1).to(device) ## num_pixels x 2
            pixel_coords2 = torch.from_numpy(batch_data_samples1[i].pixel_coords2).to(device) ## num_pixels x 2

            assert pixel_coords1.shape[0] == pixel_coords2.shape[0]

            if len(pixel_coords1) > self.max_pixels:
                idx = torch.randperm(len(pixel_coords1))[:self.max_pixels]
                pixel_coords1 = pixel_coords1[idx]
                pixel_coords2 = pixel_coords2[idx]

            desc1 = desc1[:, pixel_coords1[:, 1], pixel_coords1[:, 0]].T ## num_pixels x 32, N x D
            desc2 = desc2[:, pixel_coords2[:, 1], pixel_coords2[:, 0]].T ## num_pixels x 32, N x D

            if len(desc1) == 0:
                continue

            num_correspondences += len(desc1)

            ## tempered similarities
            desc1 = desc1.unsqueeze(0) ## 1 x N x D
            desc2 = desc2.unsqueeze(0) ## 1 x N x D

            sim = self.get_similarities(desc1, desc2, euc=False) / self.temperature ## 1 x N x N, cosine similarities
            sim[sim.isnan()] = -torch.inf  # ignore nans
            sim = sim.exp_()  # save peak memory
            positives = sim.diagonal(dim1=-2, dim2=-1)
            this_loss = -(torch.log((positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps))) ## dual softmax infoNCE, 1 x N
            loss += this_loss.sum() / (len(desc1) + self.eps)

        reduction = reduction_override if reduction_override else self.reduction

        if num_correspondences == 0:
            return torch.tensor(loss, device=device) * self.loss_weight

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight

        return loss

    @property
    def loss_name(self):
        """Returns the name of this loss function."""
        return self._loss_name
