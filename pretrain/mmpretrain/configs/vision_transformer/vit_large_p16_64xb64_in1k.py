# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmpretrain.models import CrossEntropyLoss, Mixup

with read_base():
    from .._base_.datasets.imagenet_bs64_pil_resize_autoaug import *
    from .._base_.default_runtime import *
    from .._base_.models.vit_base_p16 import *
    from .._base_.schedules.imagenet_bs4096_adamw import *

# model setting
model.update(
    backbone=dict(arch='l'),
    head=dict(
        hidden_dim=3072,
        in_channels=1024,
        topk=(1, 5),
    ),
    train_cfg=dict(augments=dict(type=Mixup, alpha=0.2)),
)

model.head.loss = dict(type=CrossEntropyLoss, loss_weight=1.0)

# schedule setting
optim_wrapper.update(clip_grad=dict(max_norm=1.0))
