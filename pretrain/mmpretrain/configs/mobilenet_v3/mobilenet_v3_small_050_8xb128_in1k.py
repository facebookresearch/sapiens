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
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.optim import RMSprop

# model settings
model.merge(
    dict(
        backbone=dict(
            arch='small_050',
            norm_cfg=dict(type=BatchNorm2d, eps=1e-5, momentum=0.1)),
        head=dict(in_channels=288),
    ))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(
        type=AutoAugment,
        policies='imagenet',
        hparams=dict(pad_val=[round(x) for x in [103.53, 116.28, 123.675]])),
    dict(
        type=RandomErasing,
        erase_prob=0.2,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader

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

train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
