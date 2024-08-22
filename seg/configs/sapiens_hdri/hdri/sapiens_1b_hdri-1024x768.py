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
num_epochs=200
num_tokens = (image_size[0] // patch_size) * (image_size[1] // patch_size)

pretrained_checkpoint='../pretrain/checkpoints/shutterstock_instagram/sapiens_1b/sapiens_1b_shutterstock_instagram_epoch_173_clean.pth'

vis_every_iters=100
# vis_every_iters=1

evaluate_every_n_epochs = 1
save_every_n_epochs = 1
# save_every_n_epochs = 100

##--------------------------------------------------------------------------
norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, ## convert from bgr to rgb for pretrained models
    pad_val=0,
    size=(image_size[1], image_size[0]),
    seg_pad_val=255)
model = dict(
    type='HDRIEstimator',
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
        type='VitHDRIHead',
        in_channels=embed_dim,
        channels=768,
        deconv_out_channels=None,
        deconv_kernel_sizes=None,
        conv_out_channels=(768, 768, 768),
        conv_kernel_sizes=(1, 1, 1),
        final_layer=((num_tokens // ((2*2*2)*(2*2*2))) * 768, 1536, 1536, 1536, 1536), ## 64 = 8 x 8, ie three conv layers, downsample by 2 each. three mlps. out is 1536 = 16 x 32 x 3
        num_classes=-1,
        hdri_size=(16, 32), ## height x width
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[\
        dict(type='L1Loss', reduction='mean', loss_weight=1.0, eps=-1)],
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
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=save_every_n_epochs, max_keep_ckpts=-1),
    visualization=dict(type='HDRIVisualizationHook', interval=vis_every_iters, max_samples=4, vis_image_width=384, vis_image_height=512),
    )

##-----------------------------------------------------------------
# dataset settings
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='HDRIResize', scale=(768, 1024)), ## in case if image was too small and random crop returned the original image
    dict(type='GenerateHDRITarget'), ## this should be less than used in the loss functions
    dict(type='PackHDRIInputs')
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='Resize', scale=(768, 1024), keep_ratio=False), ## this is width x height, 768 x 1024
    dict(type='LoadAnnotations'),
    dict(type='PackHDRIInputs')
]

##------------------------------------------------------------------------
dataset = dict(
        type='HDRIDataset',
        data_root='/ablation_transient_a/junxuanli/CARE_urca2/CARE_universal_ca2_gsplat_relight/runs/dataset_version_3',
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
        type='NormalCombinedDataset', #
        metainfo=dict(from_file='configs/_base_/datasets/hdri.py'),
        datasets=train_datasets,
        pipeline=train_pipeline))


##-----------------------------------------------------------------
val_evaluator = None
test_evaluator = None
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
