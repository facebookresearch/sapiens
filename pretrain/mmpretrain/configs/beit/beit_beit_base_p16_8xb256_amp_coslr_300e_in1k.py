# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from mmengine.dataset import DefaultSampler, default_collate
from mmengine.hooks import CheckpointHook
from mmengine.model import ConstantInit, PretrainedInit, TruncNormalInit
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.datasets import (BEiTMaskGenerator, ColorJitter, ImageNet,
                                 LoadImageFromFile, PackInputs, RandomFlip,
                                 RandomResizedCropAndInterpolationWithTwoPic)
from mmpretrain.models import (BEiT, BEiTPretrainViT, BEiTV1Head,
                               CrossEntropyLoss, DALLEEncoder,
                               TwoNormDataPreprocessor)

# dataset settings
dataset_type = ImageNet
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type=TwoNormDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    second_mean=[-31.875, -31.875, -31.875],
    second_std=[318.75, 318.75, 318.75],
    to_rgb=True)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ColorJitter, brightness=0.4, contrast=0.4, saturation=0.4,
        hue=0.),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(
        type=RandomResizedCropAndInterpolationWithTwoPic,
        size=224,
        second_size=112,
        interpolation='bicubic',
        second_interpolation='lanczos',
        scale=(0.08, 1.0)),
    dict(
        type=BEiTMaskGenerator,
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=None,
        min_num_patches=16),
    dict(type=PackInputs)
]
train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# model settings
model = dict(
    type=BEiT,
    backbone=dict(
        type=BEiTPretrainViT,
        arch='base',
        patch_size=16,
        drop_path_rate=0.1,
        final_norm=True,
        out_type='raw',
        layer_scale_init_value=0.1,
        init_cfg=[
            dict(type=TruncNormalInit, std=0.02, layer='Linear'),
            dict(type=TruncNormalInit, std=0.02, layer='Conv2d'),
            dict(type=ConstantInit, layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=None,
    head=dict(
        type=BEiTV1Head,
        embed_dims=768,
        num_embed=8192,
        loss=dict(type=CrossEntropyLoss)),
    target_generator=dict(
        type=DALLEEncoder,
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/dalle_encoder.pth',  # noqa: E501
        )))

# optimizer wrapper
optim_wrapper = dict(
    type=AmpOptimWrapper,
    loss_scale='dynamic',
    optimizer=dict(
        type=AdamW, lr=1.5e-3, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=3.0),
    paramwise_cfg=dict(
        custom_keys={
            # the following configurations are designed for BEiT
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=300)
default_hooks.update(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3))

randomness.update(seed=0, diff_rank_seed=True)

find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
