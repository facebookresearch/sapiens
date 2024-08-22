# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.model import TruncNormalInit

from mmpretrain.models import (ConvNeXt, CutMix, ImageClassifier,
                               LabelSmoothLoss, LinearClsHead, Mixup)

# Model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=ConvNeXt, arch='base', drop_path_rate=0.5),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=1024,
        loss=dict(type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
    init_cfg=dict(
        type=TruncNormalInit, layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0),
    ]),
)
