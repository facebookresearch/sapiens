# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

_base_ = ['../../_base_/default_runtime.py']


##-----------------------------------------------------------------
model_name = 'sapiens_0.3b'; embed_dim=1024; num_layers=24
# model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens_2b'; embed_dim=1920; num_layers=48

pretrained_checkpoint='../pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth'

##-----------------------------------------------------------------
# evaluate_every_n_epochs = 10 ## default
evaluate_every_n_epochs = 1

vis_every_iters=100
# vis_every_iters=1

image_size = [768, 1024] ## width x height
sigma = 6 ## sigma is 2 for 256
scale = 4
patch_size=16
num_keypoints=308
num_epochs=210 ## default 210

# runtime
train_cfg = dict(max_epochs=num_epochs, val_interval=evaluate_every_n_epochs)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

## make sure the num_layers is same as the architecture
optim_wrapper = dict(
    optimizer=dict(
        # type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
        type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.85,
        # layer_decay_rate=0.9,
        # layer_decay_rate=0.95,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2), ## default
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=num_epochs,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512) ## default not enabled
# auto_scale_lr = dict(base_batch_size=512, enable=True) ## enables. Will change LR based on actual batch size this base batch size

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='goliath/AP', rule='greater', by_epoch=False, interval=1000, max_keep_ckpts=-1),
    visualization=dict(type='GeneralPoseVisualizationHook', enable=True, interval=vis_every_iters, scale=scale),
    logger=dict(type='LoggerHook', interval=10),
    )

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/scale), int(image_size[1]/scale)), sigma=sigma) ## sigma is 2 for 256

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
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
    head=dict(
        type='HeatmapHead',
        in_channels=embed_dim,
        out_channels=num_keypoints,
        deconv_out_channels=(768, 768), ## this will 2x at each step. so total is 4x
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 768),
        conv_kernel_sizes=(1, 1),
        # loss=dict(type='KeypointMSELoss', use_target_weight=True),
        loss=dict(type='KeypointOHKMMSELoss', use_target_weight=True, topk=128), ## loss only for top 64 keypoints. for finetuning later.
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
    ))

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'), ## default prob is 0.5
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

###---------------------------------------------------------------
# datasets
dataset_goliath_train = dict(
        type='GoliathDataset',
        data_root='data/goliath',
        data_mode='topdown',
        serialize_data=False,
        ann_file='goliath_keypoint_344_train:2023082601.json',
    )

###---------------------------------------------------------------
train_datasets = [dataset_goliath_train]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/goliath.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
    ),
    )

###------------------------------------------------------------
dataset_goliath_val=dict(
        type='GoliathEvalDataset',
        # data_root='data/goliath/test_10000',
        data_root='data/goliath/test_5000',
        data_mode='topdown',
        serialize_data=False,
        ann_file='annotations/person_keypoints_test2023.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
    )

bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_70_person.json'
dataset_coco_wholebody_val=dict(
        type='CocoWholeBodyDataset',
        data_root='data/coco',
        data_mode='topdown',
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        bbox_file=bbox_file,
    )

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/goliath.py'),
        datasets=[dataset_goliath_val, dataset_coco_wholebody_val],
        pipeline=val_pipeline,
    ))


test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='GoliathCocoWholeBodyMetric',
    ann_file='data/goliath/test_5000/annotations/person_keypoints_test2023.json',
    coco_wholebody_ann_file='data/coco/annotations/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator
