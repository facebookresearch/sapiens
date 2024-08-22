# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

from mmpretrain.datasets import (CenterCrop, LoadImageFromFile, PackInputs,
                                 RandomFlip, RandomResizedCrop, ResizeEdge)
from mmpretrain.models import CrossEntropyLoss

with read_base():
    from .._base_.datasets.imagenet_bs64_pil_resize import *
    from .._base_.default_runtime import *
    from .._base_.models.vit_base_p16 import *
    from .._base_.schedules.imagenet_bs4096_adamw import *

# model setting
model.update(
    backbone=dict(arch='l', img_size=384),
    head=dict(in_channels=1024, topk=(1, 5)))

model.head.loss = dict(type=CrossEntropyLoss, loss_weight=1.0)

# dataset setting
data_preprocessor.update(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=384, backend='pillow'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=ResizeEdge, scale=384, edge='short', backend='pillow'),
    dict(type=CenterCrop, crop_size=384),
    dict(type=PackInputs),
]

train_dataloader.update(dataset=dict(pipeline=train_pipeline))
val_dataloader.update(dataset=dict(pipeline=test_pipeline))
test_dataloader.update(dataset=dict(pipeline=test_pipeline))

# schedule setting
optim_wrapper.update(clip_grad=dict(max_norm=1.0))
