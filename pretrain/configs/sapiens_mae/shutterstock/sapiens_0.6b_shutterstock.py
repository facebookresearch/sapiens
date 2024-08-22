_base_ = [
    '../../_base_/models/mae_vit-base-p16.py',
    '../../_base_/default_runtime.py',
]

patch_size=16
image_size=1024

## debug
# vis_every_iters=1
# save_every_epochs=1

## run
vis_every_iters=100
# save_every_epochs=1

save_every_iters=2000

# model_name = 'sapiens_0.3b'; embed_dim=1024; num_layers=24
model_name = 'sapiens_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens_2b'; embed_dim=1920; num_layers=48
# model_name = 'sapiens_4b'; embed_dim=2432; num_layers=56
# model_name = 'sapiens_8b'; embed_dim=3264; num_layers=64

num_patches=(image_size//patch_size)**2

##----------------------------------------------------------------------
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=image_size,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]

shutterstock_dataset = dict(
        type='Shutterstock',
        data_root='data/shutterstock',
        airstore_id='codec_avatar_shutterstock_image_editorial_model_v3_no_user_data',
        split='train',
        pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=512,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), ## use when indexing
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CombinedDataset',
        datasets=[shutterstock_dataset]
    ))

##----------------------------------------------------------------------
# model settings
model = dict(
    backbone=dict(type='MAEViT', arch=model_name, patch_size=patch_size, img_size=image_size, final_norm=True),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_patches=num_patches),
    head=dict(patch_size=patch_size))

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=True), ## clip gradients
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=40,
        by_epoch=True,
        begin=20,
        end=50,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)

default_hooks = dict(
    # checkpoint=dict(type='CheckpointHook', interval=save_every_epochs, max_keep_ckpts=-1), 
    checkpoint=dict(type='CheckpointHook', interval=save_every_iters, by_epoch=False, max_keep_ckpts=-1), 

    # print log every 10 iterations.
    logger=dict(type='LoggerHook', interval=10),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=True),
    )

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=4096) ## default is not on
auto_scale_lr = dict(base_batch_size=4096, enable=True)

custom_hooks = [
    dict(
        type='PretrainVisualizationHook',
        enable=True,
        vis_every_iters=vis_every_iters,
        vis_max_samples=16,
        )
]

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    # cudnn_benchmark=False, ##default
    cudnn_benchmark=True,

    # set multi process parameters
    # mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), ##default
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
