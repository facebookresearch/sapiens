# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner.loops import EpochBasedTrainLoop

from mmpretrain.engine.optimizers.lars import LARS

# optimizer wrapper
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=LARS, lr=4.8, weight_decay=1e-6, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, T_max=190, by_epoch=True, begin=10, end=200)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=200)
