# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.models.mobilenet_v3_small import *
    from .._base_.datasets.cifar10_bs16 import *
    from .._base_.schedules.cifar10_bs128 import *
    from .._base_.default_runtime import *

from mmengine.optim import MultiStepLR

# model settings
model.merge(
    dict(
        head=dict(
            _delete_=True,
            type=StackedLinearClsHead,
            num_classes=10,
            in_channels=576,
            mid_channels=[1280],
            act_cfg=dict(type=Hardswish),
            loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
            topk=(1, 5))))
# schedule settings
param_scheduler.merge(
    dict(
        type=MultiStepLR,
        by_epoch=True,
        milestones=[120, 170],
        gamma=0.1,
    ))

train_cfg.merge(dict(by_epoch=True, max_epochs=200))
