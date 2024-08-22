_base_ = [
    '../../_base_/default_runtime.py',
]

##-----------------------------------------------------------------
model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40

##-----------------------------------------------------------------
image_size = (768, 1024) ## width x height
data_preprocessor = dict(size=image_size)

patch_size=16
num_epochs=200
num_tokens = (image_size[0] // patch_size) * (image_size[1] // patch_size)

pretrained_checkpoint='../pretrain/checkpoints/shutterstock_instagram/sapiens_1b/sapiens_1b_shutterstock_instagram_epoch_173_clean.pth'

vis_every_iters=100
# vis_every_iters=1

evaluate_every_n_epochs = 1
save_every_n_epochs = 1
# save_every_n_epochs = 100

##--------------------------------------------------------------------------
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='StereoPointmapDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, ## convert from bgr to rgb for pretrained models
    pad_val=0,
    size=(image_size[1], image_size[0]),
    seg_pad_val=255)
model = dict(
    type='StereoPointmapEstimator',
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_checkpoint),
    ),
    embed_dims=embed_dim,
    num_layers=6,
    num_heads=24,
    decode_head1=dict(
        type='VitStereoPointmapHead',
        in_channels=embed_dim,
        channels=512,
        deconv_out_channels=None,
        deconv_kernel_sizes=None,
        upsample_conv_out_channels=(512, 512, 512), ## this will 2x at each step. so total is 8x
        upsample_conv_kernel_sizes=(3, 3, 3),
        conv_out_channels=(512, 512),
        conv_kernel_sizes=(1, 1),
        num_classes=3,
        dropout_ratio=0.0,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[\
        dict(type='PointmapL1Loss', reduction='mean', loss_weight=1.0, eps=-100, normalize=False),
        dict(type='PointmapSiLogLoss', loss_weight=1.0, eps=-100, background_val=-1000),
        dict(type='PointmapConsistencyLoss', loss_weight=10.0, eps=-100),
        ],
    ),
    decode_head2=dict(
        type='VitStereoPointmapHead',
        in_channels=embed_dim,
        channels=512,
        deconv_out_channels=None,
        deconv_kernel_sizes=None,
        upsample_conv_out_channels=(512, 512, 512), ## this will 2x at each step. so total is 8x
        upsample_conv_kernel_sizes=(3, 3, 3),
        conv_out_channels=(512, 512),
        conv_kernel_sizes=(1, 1),
        num_classes=3,
        dropout_ratio=0.0,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[\
        dict(type='PointmapL1Loss', reduction='mean', loss_weight=1.0, eps=-100, normalize=False),
        dict(type='PointmapSiLogLoss', loss_weight=1.0, eps=-100, background_val=-1000),
        ],
    ),
    loss_stereo_decode=dict(type='StereoPointmapCorrespondenceLoss', loss_weight=100.0),
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
        # type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
        type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.9,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='StereoPointmapLayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, begin=0, end=400,
        by_epoch=False),  # warm-up
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=num_epochs,
        by_epoch=True,
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
    # checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=save_every_n_epochs, max_keep_ckpts=-1),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=-1),
    visualization=dict(type='StereoPointmapVisualizationHook', interval=vis_every_iters, max_samples=4, vis_image_width=512, vis_image_height=512),
    )

##-----------------------------------------------------------------
# dataset settings
train_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='RandomResize',
        scale=[(384, 512), (1536, 2048)], ## width, height
        ratio_range=(0.2, 2.0),
        keep_ratio=False, ## keep the aspect ratio
    ),
    dict(type='RandomPointmapResizeCompensate'), ## compensate for the random resize augmentation
    dict(type='RandomPointmapCrop', crop_size=(1024, 768)), ## height, width. works for normal as well
    dict(type='PointmapResize', scale=(768, 1024)), ## in case if image was too small and random crop returned the original image
    dict(type='PointmapRandomFlip', prob=0.5,),
    dict(type='GeneratePointmapTarget', background_val=-1000), ## this should be less than used in the loss functions
    dict(type='GenerateStereoPointmapCorrespondences', min_overlap_percentage=2, distance_threshold=0.1),
    dict(type='StereoPointmapTransformSecondaryToAnchor', background_val=-1000),
    dict(type='PackStereoPointmapInputs', meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'K', 'M', 'pixel_coords1', 'pixel_coords2', 'overlap_percentage'))
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='Resize', scale=(768, 1024), keep_ratio=False), ## this is width x height, 768 x 1024
    dict(type='LoadAnnotations'),
    dict(type='TestPackStereoPointmapInputs', meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'padding_size', 'K', 'M'))
]

##------------------------------------------------------------------------
dataset = dict(
        type='StereoPointmapRenderPeopleDataset',
        data_root='data/render_people/synthetic',
        serialize_data=False,
        )

##------------------------------------------------------------------------
train_datasets = [dataset]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='StereoPointmapCombinedDataset',
        metainfo=None,
        datasets=train_datasets,
        pipeline=train_pipeline))


##-----------------------------------------------------------------
val_evaluator = None
test_evaluator = None
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
