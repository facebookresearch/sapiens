# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True)
image_size = 448

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=image_size,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(
    #     type='RandAugment',
    #     policies={{_base_.rand_increasing_policies}},
    #     num_policies=2,
    #     total_level=10,
    #     magnitude_level=9,
    #     magnitude_std=0.5,
    #     hparams=dict(
    #         pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
    #         interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(image_size, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=image_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root='data/imagenet',
        split='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root='data/imagenet',
        split='val',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_root='data/imagenet',
        split='val',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='accuracy')
