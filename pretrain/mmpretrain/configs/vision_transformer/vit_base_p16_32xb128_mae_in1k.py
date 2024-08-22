# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base
from mmengine.model import ConstantInit, TruncNormalInit
from torch.optim import AdamW

from mmpretrain.engine import EMAHook
from mmpretrain.models import CutMix, Mixup

with read_base():
    from .._base_.datasets.imagenet_bs64_swin_224 import *
    from .._base_.default_runtime import *
    from .._base_.models.vit_base_p16 import *
    from .._base_.schedules.imagenet_bs1024_adamw_swin import *

model.update(
    backbone=dict(drop_rate=0, drop_path_rate=0.1, init_cfg=None),
    head=dict(loss=dict(mode='original')),
    init_cfg=[
        dict(type=TruncNormalInit, layer='Linear', std=.02),
        dict(type=ConstantInit, layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(
        augments=[dict(type=Mixup, alpha=0.8),
                  dict(type=CutMix, alpha=1.0)]))

# dataset settings
train_dataloader.update(batch_size=128)

# schedule settings
optim_wrapper.update(
    optimizer=dict(
        type=AdamW,
        lr=1e-4 * 4096 / 256,
        weight_decay=0.3,
        eps=1e-8,
        betas=(0.9, 0.95)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))

# runtime settings
custom_hooks = [dict(type=EMAHook, momentum=1e-4)]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr.update(base_batch_size=4096)
