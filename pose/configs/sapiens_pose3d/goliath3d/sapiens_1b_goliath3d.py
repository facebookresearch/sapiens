_base_ = ['../../_base_/default_runtime.py']

##-----------------------------------------------------------------
# model_name = 'sapiens_0.3b'; embed_dim=1024; num_layers=24
# model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens_2b'; embed_dim=1920; num_layers=48
# model_name = 'sapiens_4b'; embed_dim=2432; num_layers=56
# model_name = 'sapiens_8b'; embed_dim=3264; num_layers=64

pretrained_checkpoint='../pretrain/checkpoints/shutterstock_instagram/sapiens_1b/sapiens_1b_shutterstock_instagram_epoch_173_clean.pth' ## shutterstock only

##-----------------------------------------------------------------
# evaluate_every_n_epochs = 10 ## default
evaluate_every_n_epochs = 1

vis_every_iters=100
# vis_every_iters=1

save_every_iters=1000

image_size = [768, 1024] ## width x height
sigma = 6 ## sigma is 2 for 256
image_to_heatmap_ratio = 4
patch_size=16
num_keypoints=116 ## without teeth 308, total 344
num_epochs=210 ## default 210
num_tokens = (image_size[0] // patch_size) * (image_size[1] // patch_size)

# runtime
train_cfg = dict(max_epochs=num_epochs, val_interval=evaluate_every_n_epochs)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

## make sure the num_layers is same as the architecture
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        # layer_decay_rate=0.9,
        layer_decay_rate=0.85,
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
    checkpoint=dict(save_best='goliath3d/AP', rule='greater', by_epoch=False, interval=save_every_iters, max_keep_ckpts=-1),
    visualization=dict(type='Pose3dVisualizationHook', enable=True, interval=vis_every_iters, scale=image_to_heatmap_ratio),
    logger=dict(type='LoggerHook', interval=10),
    )

codec = dict(type='UDPHeatmap', input_size=(image_size[0], image_size[1]), heatmap_size=(int(image_size[0]/image_to_heatmap_ratio), int(image_size[1]/image_to_heatmap_ratio)), sigma=sigma) ## sigma is 2 for 256

## visualize
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer_3d = dict(type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer_3d') ## do not remove. used during inference
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# model settings
model = dict(
    type='Pose3dTopdownEstimator',
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
        # loss=dict(type='KeypointOHKMMSELoss', use_target_weight=True, topk=128), ## loss only for top 128 keypoints. for finetuning later.
        loss=dict(type='KeypointMSELoss', use_target_weight=True), ## loss on all keypoints. f
        decoder=codec),
    pose3d_head=dict(
        type='Pose3dHeatmapHead',
        in_channels=embed_dim,
        num_keypoints=num_keypoints,
        image_to_heatmap_ratio=image_to_heatmap_ratio,
        depth_conv_out_channels=(embed_dim, embed_dim, embed_dim),
        depth_conv_kernel_sizes=(1, 1, 1),
        ## 64 = 8 x 8, ie three conv layers, downsample by 2 each. stride for conv is set to 2. three mlps. out is 308 (num_keypoints)
        depth_final_layer=((num_tokens // ((2*2*2)*(2*2*2))) * embed_dim, embed_dim, embed_dim, embed_dim, num_keypoints), 
        K_conv_out_channels=(embed_dim, embed_dim, embed_dim),
        K_conv_kernel_sizes=(1, 1, 1),
        K_final_layer=((num_tokens // ((2*2*2)*(2*2*2))) * embed_dim, embed_dim, embed_dim, embed_dim, num_keypoints*4), ## fx, fy, cx, cy
        loss=[
            # dict(type='Pose3d_L1_Loss', use_target_weight=True, loss_weight=1.0), 
            dict(type='Pose3d_Depth_L1_Loss', use_target_weight=True, loss_weight=1.0), 
            dict(type='Pose3d_K_Loss', image_width=image_size[0], image_height=image_size[1], use_target_weight=True, loss_weight=1.0), 
            dict(type='Pose3d_RelativeDepth_Loss', use_target_weight=True, loss_weight=1.0), 
            # dict(type='Pose3d_Pose2d_L1_Loss', image_width=image_size[0], image_height=image_size[1], use_target_weight=True, loss_weight=1.0), 
        ]
        ),
    test_cfg=dict(
        flip_test=False,
        shift_heatmap=False,
    ))

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='Pose3dRandomFlip', direction='horizontal'), ## default prob is 0.5
    dict(type='Pose3dRandomBBoxTransform'),
    dict(type='Pose3dTopdownAffine', input_size=codec['input_size'], use_udp=True),
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
    dict(type='Pose3dGenerateTarget'), ## 3d pose
    dict(type='PackPose3dInputs', pack_transformed=True, meta_keys=('id', 'img_id', 'img_path', 'category_id',
                            'crowd_index', 'ori_shape', 'img_shape',
                            'input_size', 'input_center', 'input_scale',
                            'flip', 'flip_direction', 'flip_indices',
                            'raw_ann_info', 'K'))
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25), ## default
    # dict(type='GetBBoxCenterScale', padding=1.0), ## as we are using the entire image. no scale padding
    dict(type='Pose3dTopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPose3dInputs', pack_transformed=True, meta_keys=('id', 'img_id', 'img_path', 'category_id',
                            'crowd_index', 'ori_shape', 'img_shape',
                            'input_size', 'input_center', 'input_scale',
                            'flip', 'flip_direction', 'flip_indices',
                            'raw_ann_info', 'K'))
]

###---------------------------------------------------------------
# datasets
dataset_goliath_train = dict(
        type='Goliath3dDataset',
        data_root='/uca/hewen/datasets',
        data_root_3d='/uca/hewen/datasets/goliath_keypoint_344_train',
        data_mode='topdown',
        serialize_data=False,
        ann_file='goliath_keypoint_344_train:2023082601.json',
    )

dataset_coco_wholebody_train = dict(
    type='CocoWholeBody2Goliath3dDataset',
    data_root='data/coco',
    data_mode='topdown',
    ann_file='annotations/coco_wholebody_train_v1.0.json',
    data_prefix=dict(img='train2017/'),
)

dataset_mpii_train = dict(
    type='Mpii2Goliath3dDataset',
    data_root='data/mpii',
    data_mode='topdown',
    ann_file='annotations/mpii_train.json',
    data_prefix=dict(img='images/'),
)

dataset_mpii_val = dict(
    type='Mpii2Goliath3dDataset',
    data_root='data/mpii',
    data_mode='topdown',
    ann_file='annotations/mpii_val.json',
    data_prefix=dict(img='images/'),
)

dataset_crowdpose_trainval = dict(
    type='Crowdpose2Goliath3dDataset',
    data_root='data/crowdpose',
    data_mode='topdown',
    ann_file='annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='images/'),
)

dataset_aic_train = dict(
    type='Aic2Goliath3dDataset',
    data_root='data/aic',
    data_mode='topdown',
    ann_file='annotations/aic_train.json',
    data_prefix=dict(img='ai_challenger_keypoint_train_20170909/'
            'keypoint_train_images_20170902/'),
)

dataset_aic_val = dict(
    type='Aic2Goliath3dDataset',
    data_root='data/aic',
    data_mode='topdown',
    ann_file='annotations/aic_val.json',
    data_prefix=dict(img='ai_challenger_keypoint_validation_20170911/'
            'keypoint_validation_images_20170911/'),
)

###---------------------------------------------------------------
external_datasets_frequency = 1 ## increase the frequency to get more samples from these datasets
external_datasets = [dataset_coco_wholebody_train, \
                dataset_mpii_train, \
                dataset_mpii_val, \
                dataset_crowdpose_trainval, \
                dataset_aic_train, \
                dataset_aic_val]

train_datasets = [dataset_goliath_train] + external_datasets_frequency * external_datasets
# train_datasets = [dataset_goliath_train] 

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/goliath3d.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
    ),
    )

###------------------------------------------------------------
dataset_goliath_val=dict(
        type='Goliath3dEvalDataset',
        data_root='data/goliath/test_5000',
        data_mode='topdown',
        serialize_data=False,
        ann_file='annotations_3d/person_keypoints_test2023.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
    )

dataset_coco_wholebody_val=dict(
        type='CocoWholeBodyDataset',
        data_root='data/coco',
        data_mode='topdown',
        ann_file='annotations/coco_wholebody_val_v1.0.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        bbox_file=None,
    )

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/goliath3d.py'),
        datasets=[dataset_goliath_val, dataset_coco_wholebody_val],
        pipeline=val_pipeline,
    ))


test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='Goliath3dCocoWholeBodyMetric',
    ann_file='data/goliath/test_5000/annotations_3d/person_keypoints_test2023.json',
    coco_wholebody_ann_file='data/coco/annotations/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator
