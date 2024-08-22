_base_ = [
    '../../_base_/default_runtime.py',
]

patch_size=16
image_size=4096

## debug
# vis_every_iters=1
# save_every_epochs=1

## run
vis_every_iters=500
save_every_iters=5000

# model_name = 'sapiens2_0.3b'; embed_dim=1024; num_layers=24
# model_name = 'sapiens2_0.6b'; embed_dim=1280; num_layers=32
# model_name = 'sapiens2_1b'; embed_dim=1536; num_layers=40
# model_name = 'sapiens2_2b'; embed_dim=1920; num_layers=48
model_name = 'sapiens2_4b'; embed_dim=2432; num_layers=56
# model_name = 'sapiens2_8b'; embed_dim=3264; num_layers=64

##----------------------------------------------------------------------
vit_image_size = image_size//4
num_patches=(vit_image_size//patch_size)**2

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
    type='MAE',
    backbone=dict(type='MAEViT2', arch=model_name, patch_size=patch_size, mask_ratio=0.75, img_size=vit_image_size, final_norm=True),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=embed_dim,
        patch_size=patch_size,
        num_patches=num_patches,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type='MAEPretrainHead2',
        norm_pix=True,
        patch_size=patch_size,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

# optimizer wrapper
optim_wrapper = dict(
    # type='AmpOptimWrapper', ## automatic mixed precision
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
    # dict(
    #     type='LinearLR',
    #     start_factor=1e-4,
    #     by_epoch=True,
    #     begin=0,
    #     end=20,
    #     convert_to_iter_based=True),
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=40,
    #     by_epoch=True,
    #     begin=20,
    #     end=50,
    #     convert_to_iter_based=True)
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        by_epoch=True,
        begin=0,
        end=50,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=save_every_iters, by_epoch=False, max_keep_ckpts=-1), 

    # print log every 10 iterations.
    logger=dict(type='LoggerHook', interval=10),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=True),
    )

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
# resume = True
resume = False

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096) ## default is not on

custom_hooks = [
    dict(
        type='Pretrain2VisualizationHook',
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


##---------------------------------------------------------------------------
# Note: this cannot be run in single gpu mode.
# FSDP sharding strategy
strategy = dict(
    type='FSDPStrategy',
    state_dict_cfg='full', ## store the full state dict in rank 0
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type='torch.distributed.fsdp.wrap.transformer_auto_wrap_policy',
            transformer_layer_cls='mmpretrain.models.backbones.TransformerEncoderLayer'),
        mixed_precision=dict(
            param_dtype='bfloat16',
            ),
        # sharding_strategy='SHARD_GRAD_OP',
        sharding_strategy='HYBRID_SHARD',
        backward_prefetch='BACKWARD_PRE',
        )
    )

# Below works but "Logger cannot be pickled"
# strategy = FSDPStrategy(
#     state_dict_cfg='full', ## store the full state dict in rank 0
#     model_wrapper=dict(
#         auto_wrap_policy=partial(
#             transformer_auto_wrap_policy,
#             transformer_layer_cls={mmpretrain.models.backbones.TransformerEncoderLayer}),
#         mixed_precision=dict(
#             param_dtype='bfloat16',
#             ),
#         sharding_strategy='FULL_SHARD',
#         )
#     )
    

# runner which supports strategies
runner_type = 'FlexibleRunner'
