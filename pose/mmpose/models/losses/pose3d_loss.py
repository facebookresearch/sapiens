# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS

@MODELS.register_module()
class Pose3d_RelativeDepth_Loss(nn.Module):
    def __init__(self, use_target_weight=False, loss_weight=1., loss_name='loss_pose3d_rel_depth'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, output, target, target_weight=None, confidence=None):
        """
        output is B x num_keypoints
        target is B x num_keypoints 
        target_weight is B x num_keypoints 
        confidence is B x num_keypoints
        """
        # Penalty for negative values # This is zero for positive values and |output| for negative values
        negative_penalty = F.relu(-output)  # B x num_keypoints

        pred_max_Z = torch.max(output, dim=1, keepdim=True)[0] ## B x 1
        pred_relative_Z = output / torch.clamp(pred_max_Z, min=1e-6) ## B x 308

        gt_max_Z = torch.max(target, dim=1, keepdim=True)[0] ## B x 1
        gt_relative_Z = target / torch.clamp(gt_max_Z, min=1e-6) ## B x 308

        loss = F.l1_loss(pred_relative_Z, gt_relative_Z, reduction='none') ## B x 308

        if confidence is not None:
            loss = loss * confidence
            negative_penalty = negative_penalty * confidence

        if self.use_target_weight:
            assert target_weight is not None
            loss = loss * target_weight ## B x 308
            loss = loss.sum() / target_weight.sum().clamp(min=1)

            negative_penalty = negative_penalty * target_weight ## B x 308
            negative_penalty = negative_penalty.sum() / target_weight.sum().clamp(min=1)
        else:
            loss = loss.mean()
            negative_penalty = negative_penalty.mean()

        return loss * self.loss_weight + negative_penalty * self.loss_weight

@MODELS.register_module()
class Pose3d_Pose2d_L1_Loss(nn.Module):
    def __init__(self, use_target_weight=False, image_width=768, image_height=1024, loss_weight=1., loss_name='loss_pose3d_pose2d_l1'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, output, target, target_weight=None, confidence=None):
        """
        output is B x num_keypoints x 2
        target is B x num_keypoints x 2
        target_weight is B x num_keypoints 
        """
        ## normalize coordinates to 0 to 1
        output_normalized = torch.stack([
            output[:, :, 0] / self.image_width,  # x is 0 to 1
            output[:, :, 1] / self.image_height  # y is 0 to 1
        ], dim=-1)

        target_normalized = torch.stack([
            target[:, :, 0] / self.image_width,  # x is 0 to 1
            target[:, :, 1] / self.image_height  # y is 0 to 1
        ], dim=-1)

        loss = F.l1_loss(output_normalized, target_normalized, reduction='none')  # B x num_keypoints x 2

        if confidence is not None:
            loss = loss * confidence.unsqueeze(-1)

        if self.use_target_weight:
            assert target_weight is not None
            target_weight = target_weight.unsqueeze(-1)  # B x num_keypoints x 1 
            loss = loss * target_weight ## B x 308
            loss = loss.sum() / (target_weight.sum() * target.shape[-1]).clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight

@MODELS.register_module()
class Pose3d_Depth_L1_Loss(nn.Module):
    def __init__(self, use_target_weight=False, loss_weight=1., loss_name='loss_pose3d_depth_l1'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, output, target, target_weight=None, confidence=None):
        """
        output is B x num_keypoints 
        target is B x num_keypoints 
        target_weight is B x num_keypoints 
        """
        loss = F.l1_loss(output, target, reduction='none') ## B x num_keypoints 

        if confidence is not None:
            loss = loss * confidence.unsqueeze(-1)

        if self.use_target_weight:
            assert target_weight is not None
            loss = loss * target_weight # B x num_keypoints x 3
            loss = loss.sum() / (target_weight.sum() * target.shape[-1]).clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight

@MODELS.register_module()
class Pose3d_L1_Loss(nn.Module):
    def __init__(self, use_target_weight=False, loss_weight=1., loss_name='loss_pose3d_l1'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, output, target, target_weight=None, confidence=None):
        """
        output is B x num_keypoints x 3
        target is B x num_keypoints x 3
        target_weight is B x num_keypoints 
        """
        loss = F.l1_loss(output, target, reduction='none') ## B x num_keypoints x 3

        if confidence is not None:
            loss = loss * confidence.unsqueeze(-1)

        if self.use_target_weight:
            assert target_weight is not None
            target_weight = target_weight.unsqueeze(-1)  # B x num_keypoints x 1 
            loss = loss * target_weight # B x num_keypoints x 3
            loss = loss.sum() / (target_weight.sum() * target.shape[-1]).clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight

@MODELS.register_module()
class Pose3d_K_Loss(nn.Module):
    def __init__(self, use_target_weight=False, image_width=768, image_height=1024,  loss_weight=1., loss_name='loss_pose3d_K'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        self.image_width = image_width
        self.image_height = image_height
        return

    def forward(self, output, target, target_weight=None, confidence=None):
        """
        output is B x num_keypoints x 4
        target is B x num_keypoints x 4
        target_weight is B x num_keypoints 
        """

        # Normalize the output and target inline
        output_normalized = torch.stack([
            output[..., 0] / self.image_width,   # fx
            output[..., 1] / self.image_height,  # fy
            output[..., 2] / self.image_width,   # cx
            output[..., 3] / self.image_height   # cy
        ], dim=-1)

        target_normalized = torch.stack([
            target[..., 0] / self.image_width,   # fx
            target[..., 1] / self.image_height,  # fy
            target[..., 2] / self.image_width,   # cx
            target[..., 3] / self.image_height   # cy
        ], dim=-1)

        # Compute the loss
        loss = F.l1_loss(output_normalized, target_normalized, reduction='none')  #

        if confidence is not None:
            loss = loss * confidence.unsqueeze(-1)

        if self.use_target_weight:
            assert target_weight is not None
            target_weight = target_weight.unsqueeze(-1)  # B x num_keypoints x 1 
            loss = loss * target_weight # B x num_keypoints x 4
            loss = loss.sum() / (target_weight.sum() * target.shape[-1]).clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class Pose3d_Confidence_Loss(nn.Module):
    def __init__(self, use_target_weight=False, loss_weight=1., loss_name='loss_pose3d_confidence'):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.loss_name = loss_name
        return

    def forward(self, confidence, target_weight=None):
        """
        confidence is B x num_keypoints
        target_weight is B x num_keypoints 
        """

        # Penalize low confidence predictions
        loss = -torch.log(confidence + 1e-6)  # Add small epsilon to prevent log(0)

        if self.use_target_weight:
            assert target_weight is not None
            loss = loss * target_weight
            loss = loss.sum() / target_weight.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss * self.loss_weight
