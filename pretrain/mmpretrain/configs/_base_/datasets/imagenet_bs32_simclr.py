# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmcv.transforms import (LoadImageFromFile, RandomApply, RandomFlip,
                             RandomGrayscale)
from mmengine.dataset import DefaultSampler, default_collate

from mmpretrain.datasets import (ColorJitter, GaussianBlur, ImageNet,
                                 MultiView, PackInputs, RandomResizedCrop)
from mmpretrain.models import SelfSupDataPreprocessor

# dataset settings
dataset_type = ImageNet
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type=SelfSupDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

view_pipeline = [
    dict(type=RandomResizedCrop, scale=224, backend='pillow'),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomApply,
        transforms=[
            dict(
                type=ColorJitter,
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(
        type=RandomGrayscale,
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type=GaussianBlur,
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
]

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=MultiView, num_views=2, transforms=[view_pipeline]),
    dict(type=PackInputs)
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=ImageNet,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
