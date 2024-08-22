# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.config import read_base

with read_base():
    from .._base_.datasets.mot_challenge import *
    from .qdtrack_faster_rcnn_r50_fpn_4e_base import *

from mmdet.evaluation import CocoVideoMetric, MOTChallengeMetric

# evaluator
val_evaluator = [
    dict(type=CocoVideoMetric, metric=['bbox'], classwise=True),
    dict(type=MOTChallengeMetric, metric=['HOTA', 'CLEAR', 'Identity'])
]
