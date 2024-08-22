# dataset settings
dataset_type = 'InShop'
data_preprocessor = dict(
    num_classes=3997,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=512),
    dict(type='RandomCrop', crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=512),
    dict(type='CenterCrop', crop_size=448),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/inshop',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

query_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/inshop',
        split='query',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

gallery_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/inshop',
        split='gallery',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_dataloader = query_dataloader
val_evaluator = [
    dict(type='RetrievalRecall', topk=1),
    dict(type='RetrievalAveragePrecision', topk=10),
]

test_dataloader = val_dataloader
test_evaluator = val_evaluator
