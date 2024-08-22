# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from ..._base_.datasets.imagenet_bs64_swin_224 import *
    from ..._base_.schedules.imagenet_bs1024_adamw_swin import *
    from ..._base_.default_runtime import *

from mmengine.hooks import CheckpointHook
from mmengine.model import PretrainedInit, TruncNormalInit
from mmengine.optim import CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from mmpretrain.datasets import LoadImageFromFile, PackInputs, RandomFlip
from mmpretrain.engine.optimizers import \
    LearningRateDecayOptimWrapperConstructor
from mmpretrain.models import (BEiTViT, ImageClassifier, LabelSmoothLoss,
                               LinearClsHead)
from mmpretrain.models.utils.batch_augments import CutMix, Mixup

data_preprocessor = dict(
    num_classes=1000,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=BEiTViT,
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_type='avg_featmap',
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
        init_cfg=dict(type=PretrainedInit, checkpoint='', prefix='backbone.')),
    neck=None,
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=768,
        loss=dict(type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type=TruncNormalInit, layer='Linear', std=0.02)]),
    train_cfg=dict(
        augments=[dict(type=Mixup, alpha=0.8),
                  dict(type=CutMix, alpha=1.0)]))

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=RandomResizedCrop,
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(
        type=RandAugment,
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type=RandomErasing,
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type=PackInputs)
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
    dict(type=PackInputs)
]

train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=128, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(type=AdamW, lr=4e-3, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor=LearningRateDecayOptimWrapperConstructor,
    paramwise_cfg=dict(
        _delete_=True,
        layer_decay_rate=0.65,
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
        end=20,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        by_epoch=True,
        begin=20,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=2))

train_cfg = dict(by_epoch=True, max_epochs=100)

randomness = dict(seed=0)
