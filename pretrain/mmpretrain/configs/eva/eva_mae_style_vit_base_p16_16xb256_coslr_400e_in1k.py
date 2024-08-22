# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.models.mae_vit_base_p16 import *
    from .._base_.datasets.imagenet_bs512_mae import *
    from .._base_.default_runtime import *

from mmengine.hooks import CheckpointHook
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.models import (EVA, CLIPGenerator, CosineSimilarityLoss,
                               MAEPretrainDecoder, MIMHead)

# dataset settings
train_dataloader.batch_size = 256

# model settings
model.type = EVA
model.init_cfg = None
model.backbone.update(init_cfg=[
    dict(type='Xavier', distribution='uniform', layer='Linear'),
    dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
])
model.neck.update(
    type=MAEPretrainDecoder,
    predict_feature_dim=512,
    init_cfg=[
        dict(type='Xavier', distribution='uniform', layer='Linear'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
model.head = dict(
    type=MIMHead,
    loss=dict(type=CosineSimilarityLoss, shift_factor=2.0, scale_factor=2.0))
model.target_generator = dict(
    type=CLIPGenerator,
    tokenizer_path=  # noqa
    'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/clip_vit_base_16.pth.tar'  # noqa
)

# optimizer wrapper
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))
find_unused_parameters = True

# learning rate scheduler
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        T_max=360,
        by_epoch=True,
        begin=40,
        end=400,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=400)
default_hooks.checkpoint = dict(
    type=CheckpointHook, interval=1, max_keep_ckpts=3)

randomness.update(dict(seed=0, diff_rank_seed=True))

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
