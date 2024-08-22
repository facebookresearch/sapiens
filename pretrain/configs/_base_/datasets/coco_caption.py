# data settings
# coco caption annotations can be grabbed from LAVIS repo
# https://github.com/salesforce/LAVIS/blob/main/lavis/configs/datasets/coco/defaults_cap.yaml
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
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
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='CleanCaption', keys='gt_caption'),
    dict(
        type='PackInputs',
        algorithm_keys=['gt_caption'],
        meta_keys=['image_id'],
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(384, 384),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='COCOCaption',
        data_root='data/coco',
        ann_file='annotations/coco_karpathy_train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type='COCOCaption',
        data_root='data/coco',
        ann_file='annotations/coco_karpathy_val.json',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(
    type='COCOCaption',
    ann_file='data/coco/annotations/coco_karpathy_val_gt.json',
)

# # If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
