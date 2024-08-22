# dataset settings
dataset_type = 'ImageNet'
data_root = 'data/imagenet/'
data_preprocessor = dict(
    type='TwoNormDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    second_mean=[127.5, 127.5, 127.5],
    second_std=[127.5, 127.5, 127.5],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        second_size=224,
        interpolation='bicubic',
        second_interpolation='bicubic',
        scale=(0.2, 1.0)),
    dict(
        type='BEiTMaskGenerator',
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=75,
        min_num_patches=16),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        pipeline=train_pipeline))
