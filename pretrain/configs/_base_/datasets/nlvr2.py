# dataset settings
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

train_pipeline = [
    dict(
        type='ApplyToList',
        # NLVR requires to load two images in task.
        scatter_key='img_path',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=384,
                interpolation='bicubic',
                backend='pillow'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        ],
        collate_keys=['img', 'scale_factor', 'ori_shape'],
    ),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text'],
        meta_keys=['image_id'],
    ),
]

test_pipeline = [
    dict(
        type='ApplyToList',
        # NLVR requires to load two images in task.
        scatter_key='img_path',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                scale=(384, 384),
                interpolation='bicubic',
                backend='pillow'),
        ],
        collate_keys=['img', 'scale_factor', 'ori_shape'],
    ),
    dict(
        type='PackInputs',
        algorithm_keys=['text'],
        meta_keys=['image_id'],
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='NLVR2',
        data_root='data/nlvr2',
        ann_file='dev.json',
        data_prefix='dev',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type='NLVR2',
        data_root='data/nlvr2',
        ann_file='dev.json',
        data_prefix='dev',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
