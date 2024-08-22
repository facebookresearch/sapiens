# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.model.weight_init import NormalInit
from torch.nn.modules.activation import Hardswish

from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, MobileNetV3,
                               StackedLinearClsHead)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=MobileNetV3, arch='small'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=StackedLinearClsHead,
        num_classes=1000,
        in_channels=576,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type=Hardswish),
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        init_cfg=dict(
            type=NormalInit, layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)))
