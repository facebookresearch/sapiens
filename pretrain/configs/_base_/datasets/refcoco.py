# data settings

data_preprocessor = dict(
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                backend='cv2')
        ],
        prob=0.5),
    dict(
        type='mmdet.RandomCrop',
        crop_type='relative_range',
        crop_size=(0.8, 0.8),
        allow_negative_crop=False),
    dict(
        type='RandomChoiceResize',
        scales=[(384, 384), (360, 360), (344, 344), (312, 312), (300, 300),
                (286, 286), (270, 270)],
        keep_ratio=False),
    dict(
        type='RandomTranslatePad',
        size=384,
        aug_translate=True,
    ),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes', 'scale_factor'],
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
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes', 'scale_factor'],
        meta_keys=['image_id'],
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        data_prefix='train2014',
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        data_prefix='train2014',
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='VisualGroundingMetric')

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='RefCOCO',
        data_root='data/coco',
        data_prefix='train2014',
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='testA',  # or 'testB'
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator
