# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
# Refers to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification

from mmengine.config import read_base

with read_base():
    from .._base_.models.mobilenet_v3_small import *
    from .._base_.datasets.imagenet_bs128_mbv3 import *
    from .._base_.default_runtime import *

from mmengine.optim import StepLR
from torch.optim import RMSprop

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type=RMSprop,
        lr=0.064,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-5))

param_scheduler = dict(type=StepLR, by_epoch=True, step_size=2, gamma=0.973)

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
