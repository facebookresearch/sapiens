default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), ## default
    # mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0), ## faster?
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')

##-----------------------------------------------------------------
model_name = 'sapiens2_0.3b'; embed_dim=1024; num_layers=24
# model_name = 'sapiens2_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens2_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens2_2b'; embed_dim=1920; num_layers=48

##-----------------------------------------------------------------
image_size = (3072, 4096) ## width x height
vit_image_size = (image_size[0] // 4, image_size[1] // 4)

patch_size=16
num_epochs=200

pretrained_checkpoint='/uca/rawalk/sapiens2_host/pretrain/checkpoints/sapiens2_0.3b/sapiens2_0.3b_shutterstock_iter_470000_clean.pth'

vis_every_iters=100
# vis_every_iters=1

evaluate_every_n_epochs = 1

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
    type='DepthEstimator',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.VisionTransformer2',
        arch=model_name,
        img_size=(vit_image_size[1], vit_image_size[0]),
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
    decode_head=dict(
        type='VitNormalHead2',
        in_channels=embed_dim,
        channels=768,
        deconv_out_channels=None,
        deconv_kernel_sizes=None,
        upsample_conv_out_channels=(768, 768, 768, 768), ## this will 2x at each step. so total is 16x. 1k output.
        upsample_conv_kernel_sizes=(3, 3, 3, 3),
        conv_out_channels=(768, 768, 768),
        conv_kernel_sizes=(1, 1, 1),
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[\
        dict(type='CosineSimilarityLoss', reduction='mean', loss_weight=1.0, eps=-100), 
        dict(type='L1Loss', reduction='mean', loss_weight=1.0, eps=-100),
        dict(type='EdgeAwareLoss', reduction='mean', loss_weight=1.0, eps=-100),
        dict(type='UnitNormLoss', reduction='mean', loss_weight=0.1, eps=-100),
        ],
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
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1), ## original lr
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
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=-1),
    visualization=dict(type='NormalVisualizationHook', interval=vis_every_iters, max_samples=4, vis_image_width=384, vis_image_height=512),
    )

##-----------------------------------------------------------------
# dataset settings
train_pipeline = [
    dict(type='LoadImage'),
    # dict(type='RandomBackground', prob=0.5, background_images_root='/uca/sstk/vqa_non_human'),
    dict(
        type='RandomResize',
        scale=(3072, 4096), ## width, height
        ratio_range=(0.2, 2.0), #
        keep_ratio=True),
    dict(type='RandomNormalResizeCompensate'), ## compensate for the random resize augmentation
    dict(type='RandomNormalCrop', crop_size=(4096, 3072)), ## height, width. works for normal as well
    dict(type='NormalResize', scale=(3072, 4096)), ## in case if image was too small and random crop returned the original image
    dict(type='NormalRandomFlip', prob=0.5,),
    dict(type='GenerateNormalTarget', background_val=-1000), ## this should be less than used in the loss functions
    dict(type='PhotoMetricDistortion'),
    dict(type='PackNormalInputs')
]

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='Resize', scale=(3072, 4096), keep_ratio=False), ## this is width x height, 3072 x 4096
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackNormalInputs')
]

##------------------------------------------------------------------------
dataset_train = dict(
        type='NormalRenderPeopleDataset',
        data_root='data/render_people/synthetic',
        serialize_data=False,
        )

##------------------------------------------------------------------------
train_datasets = [dataset_train]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NormalCombinedDataset', #
        metainfo=dict(from_file='configs/_base_/datasets/render_people.py'),
        datasets=train_datasets,
        pipeline=train_pipeline))

##-----------------------------------------------------------------
val_evaluator = None
test_evaluator = None
val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None

# ##---------------------------------------------------------------------------
# Note: this cannot be run in single gpu mode.
# FSDP sharding strategy
strategy = dict(
    type='FSDPStrategy',
    state_dict_cfg='full', ## store the full state dict in rank 0
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type='torch.distributed.fsdp.wrap.size_based_auto_wrap_policy',
            min_num_params=1e7),
        mixed_precision=dict(
            param_dtype='bfloat16',
            reduce_dtype='bfloat16',
            buffer_dtype='bfloat16',
            ),
        sharding_strategy='FULL_SHARD',
        use_orig_params=True  # important to set true for setting differential learning rate in encoder correctly
        )
    )

# runner which supports strategies
runner_type = 'FlexibleRunner'
