# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .mmpose_inferencer import MMPoseInferencer
from .pose2d_inferencer import Pose2DInferencer
from .pose3d_inferencer import Pose3DInferencer
from .utils import get_model_aliases

__all__ = [
    'Pose2DInferencer', 'MMPoseInferencer', 'get_model_aliases',
    'Pose3DInferencer'
]
