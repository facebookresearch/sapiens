# data settings

data_preprocessor = dict(
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys=['question', 'gt_answer']),
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=[],
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(480, 480),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys=['question', 'gt_answer']),
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=[],
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='OCRVQA',
        data_root='data/ocrvqa',
        data_prefix='images',
        ann_file='annotations/dataset.json',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type='OCRVQA',
        data_root='data/ocrvqa',
        data_prefix='images',
        ann_file='annotations/dataset.json',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='VQAAcc')

test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type='OCRVQA',
        data_root='data/ocrvqa',
        data_prefix='images',
        ann_file='annotations/dataset.json',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='VQAAcc')
