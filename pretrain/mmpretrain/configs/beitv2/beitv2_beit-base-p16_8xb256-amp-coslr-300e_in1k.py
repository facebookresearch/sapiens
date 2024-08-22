# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs256_beitv2 import *
    from .._base_.default_runtime import *

from mmengine.model import ConstantInit, PretrainedInit, TruncNormalInit
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import EpochBasedTrainLoop
from torch.optim import AdamW

from mmpretrain.models import (VQKD, BEiT, BEiTPretrainViT, BEiTV2Head,
                               BEiTV2Neck, CrossEntropyLoss)

# model settings
vqkd_encoder = dict(
    arch='base',
    img_size=224,
    patch_size=16,
    in_channels=3,
    out_indices=-1,
    drop_rate=0.,
    drop_path_rate=0.,
    norm_cfg=dict(type='LN', eps=1e-6),
    final_norm=True,
    out_type='featmap',
    with_cls_token=True,
    frozen_stages=-1,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    layer_scale_init_value=0.,
    interpolate_mode='bicubic',
    patch_cfg=dict(),
    layer_cfgs=dict(),
    init_cfg=None)

layer_scale_init_value = 0.1
drop_path_rate = 0.  # 0. for 300 epochs and 0.1 for 1600 epochs.
model = dict(
    type=BEiT,
    backbone=dict(
        type=BEiTPretrainViT,
        arch='base',
        patch_size=16,
        out_indices=[-4, -1],
        drop_path_rate=drop_path_rate,
        final_norm=False,
        out_type='raw',
        layer_scale_init_value=layer_scale_init_value,
        init_cfg=[
            dict(type=TruncNormalInit, std=0.02, layer='Linear'),
            dict(type=TruncNormalInit, std=0.02, layer='Conv2d'),
            dict(type=ConstantInit, layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=dict(
        type=BEiTV2Neck,
        num_layers=2,
        early_layers=9,
        backbone_arch='base',
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    ),
    head=dict(
        type=BEiTV2Head,
        embed_dims=768,
        num_embed=8192,
        loss=dict(type=CrossEntropyLoss)),
    target_generator=dict(
        type=VQKD,
        encoder_config=vqkd_encoder,
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint=  # noqa
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/vqkd_encoder.pth'  # noqa
        )))

# optimizer wrapper
optim_wrapper = dict(
    type=AmpOptimWrapper,
    loss_scale='dynamic',
    # betas: (0.9, 0.98) for 300 epochs and (0.9, 0.999) for 1600 epochs.
    optimizer=dict(
        type=AdamW, lr=1.5e-3, betas=(0.9, 0.98), weight_decay=0.05),
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
default_hooks = dict(
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type=CheckpointHook, interval=1, max_keep_ckpts=3))

randomness = dict(seed=0, diff_rank_seed=True)

find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
