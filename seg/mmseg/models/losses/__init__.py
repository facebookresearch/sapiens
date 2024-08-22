# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .accuracy import Accuracy, accuracy
from .boundary_loss import BoundaryLoss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .huasdorff_distance_loss import HuasdorffDisstanceLoss
from .lovasz_loss import LovaszLoss
from .ohem_cross_entropy_loss import OhemCrossEntropy
from .silog_loss import SiLogLoss
from .tversky_loss import TverskyLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

from .cosine_similarity_loss import CosineSimilarityLoss
from .l1_loss import L1Loss
from .l1_loss import MetricDepthL1Loss
from .metric_silog_loss import MetricSiLogLoss
from .pointmap_silog_loss import PointmapSiLogLoss
from .pointmap_consistency_loss import PointmapConsistencyLoss
from .pointmap_l1_loss import PointmapL1Loss
from .stereo_pointmap_l1_loss import StereoPointmapL1Loss
from .stereo_pointmap_correspondence_loss import StereoPointmapCorrespondenceLoss
from .stereo_correspondences_loss import StereoCorrespondencesLoss
from .edge_aware_loss import EdgeAwareLoss
from .unit_norm_loss import UnitNormLoss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'TverskyLoss', 'OhemCrossEntropy', 'BoundaryLoss',
    'HuasdorffDisstanceLoss', 'SiLogLoss', 'CosineSimilarityLoss',
    'L1Loss', 'MetricDepthL1Loss', 'PointmapSiLogLoss', 'PointmapConsistencyLoss',
    'MetricSiLogLoss', 'PointmapL1Loss', 'StereoPointmapL1Loss', 'StereoPointmapCorrespondenceLoss',
    'StereoCorrespondencesLoss', 'EdgeAwareLoss', 'UnitNormLoss',
]
