# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.config import read_base

from mmpretrain.engine import EMAHook

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_224 import *
    from .._base_.default_runtime import *
    from .._base_.models.convnext_base import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

# dataset setting
train_dataloader.update(batch_size=128)

# schedule setting
optim_wrapper.update(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type=EMAHook, momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr.update(base_batch_size=4096)
