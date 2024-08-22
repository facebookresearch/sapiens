# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

_base_ = [
    '../../_base_/default_runtime.py',
]

##-----------------------------------------------------------------
# model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens_2b'; embed_dim=1920; num_layers=48

##-----------------------------------------------------------------
image_size = (768, 1024) ## width x height
data_preprocessor = dict(size=image_size)

patch_size=16
num_epochs=400

pretrained_checkpoint='../pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth'

vis_every_iters=100
# vis_every_iters=1

evaluate_every_n_epochs = 1

num_classes = 19 + 1 ## add background class

CLASS_WEIGHT = [
    0.1,  # Background (very common)
    1.5,  # Hat (less common)
    1.0,  # Hair (common)
    1.5,  # Glove (less common)
    1.5,  # Sunglasses (less common)
    0.8,  # UpperClothes (common)
    1.2,  # Dress (less common)
    1.0,  # Coat (common)
    1.5,  # Socks (less common)
    0.8,  # Pants (common)
    1.2,  # Jumpsuits (less common)
    1.5,  # Scarf (less common)
    1.5,  # Skirt (less common)
    0.7,  # Face (common)
    1.0,  # Left-arm (common)
    1.0,  # Right-arm (common)
    1.0,  # Left-leg (common)
    1.0,  # Right-leg (common)
    1.5,  # Left-shoe (less common)
    1.5   # Right-shoe (less common)
]

##--------------------------------------------------------------------------
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, ## convert from bgr to rgb for pretrained models
    pad_val=0,
    size=(image_size[1], image_size[0]),
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch=model_name,
        img_size=(image_size[1], image_size[0]),
        patch_size=patch_size,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_checkpoint),
    ),
    decode_head=dict(
        type='VitHead',
        in_channels=embed_dim,
        channels=768,
        deconv_out_channels=(768, 768, 768), ## this will 2x at each step. so total is 4x
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=(768, 768, 768),
        conv_kernel_sizes=(1, 1, 1),
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, reduction='none', class_weight=CLASS_WEIGHT),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

##--------------------------------------------------------------------------
# optimizer
custom_imports = dict(
    imports=['mmseg.engine.optimizers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

## make sure the num_layers is same as the architecture
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.85,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, begin=0, end=400,
        by_epoch=False),  # warm-up
    # dict(
    #     type='PolyLR',
    #     eta_min=0.0,
    #     power=1.0,
    #     begin=0,
    #     end=num_epochs,
    #     by_epoch=True,
    # )
    dict(
        type='CosineAnnealingLR',
        T_max=num_epochs,
        by_epoch=True,
        begin=0,
        end=num_epochs
    )
]

##--------------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=num_epochs, val_interval=evaluate_every_n_epochs)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=True)

# hooks
env_cfg = dict(
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=-1),
    visualization=dict(type='GeneralSegVisualizationHook', interval=vis_every_iters, max_samples=4, vis_image_width=384, vis_image_height=512),
    )

##-----------------------------------------------------------------
# https://github.com/MengzhangLI/mmsegmentation/blob/51bdfffda45118b418366a43f9e259fd0211710c/configs/_base_/datasets/lip_321x321.py
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(768, 1024), ## width, height
        ratio_range=(0.75, 1.25),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 768), cat_max_ratio=0.75), ## height, width
    dict(type='Resize', scale=(768, 1024), keep_ratio=False), ## in case if image was too small and random crop returned the original image
    dict(type='RandomRotate', prob=0.5, degree=45, pad_val=0, seg_pad_val=255), ## the black pixels are set as 255, dont care?
    dict(type='RandomFlip', prob=0.5, swap_seg_labels=[(14, 15), (16, 17), (18, 19)]),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(768, 1024), keep_ratio=False), ## this is width x height, 768 x 1024
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

##------------------------------------------------------------------------
dataset_lip_train = dict(
        type='LIPDataset',
        data_root='data/lip',
        data_prefix=dict(img_path='images/train_images', seg_map_path='segmentations/train_segmentations'),
        )

train_datasets = [dataset_lip_train]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/lip.py'),
        datasets=train_datasets,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LIPDataset',
        data_root='data/lip',
        data_prefix=dict(img_path='images/val_images', seg_map_path='segmentations/val_segmentations'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

##-----------------------------------------------------------------
test_dataloader = val_dataloader
