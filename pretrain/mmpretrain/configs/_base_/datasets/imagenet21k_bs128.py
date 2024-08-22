# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.dataset import DefaultSampler

from mmpretrain.datasets import (ImageNet21k, LoadImageFromFile, PackInputs,
                                 RandomFlip, RandomResizedCrop)

# dataset settings
dataset_type = ImageNet21k
data_preprocessor = dict(
    num_classes=21842,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=224),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet21k',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)
