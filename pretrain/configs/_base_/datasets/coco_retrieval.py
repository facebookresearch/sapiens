# data settings
# Here are the links to download the annotations for coco retrieval for conveniency # noqa
# https://download.openmmlab.com/mmclassification/datasets/coco_retrieval/caption_karpathy_train2014.json
# https://download.openmmlab.com/mmclassification/datasets/coco_retrieval/caption_karpathy_val2014.json
# https://download.openmmlab.com/mmclassification/datasets/coco_retrieval/caption_karpathy_test2014.json
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.0)),
    dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        crop_ratio_range=(0.5, 1.0),
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        magnitude_level=5),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'is_matched'],
        meta_keys=['image_id']),
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
        algorithm_keys=['text', 'gt_text_id', 'gt_image_id'],
        meta_keys=['image_id']),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    dataset=dict(
        type='COCORetrieval',
        data_root='data/coco',
        ann_file='annotations/caption_karpathy_train2014.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=16,
    dataset=dict(
        type='COCORetrieval',
        data_root='data/coco',
        ann_file='annotations/caption_karpathy_val2014.json',
        pipeline=test_pipeline,
        # This is required for evaluation
        test_mode=True,
    ),
    sampler=dict(type='SequentialSampler', subsample_type='sequential'),
    persistent_workers=True,
)

val_evaluator = dict(type='RetrievalRecall', topk=(1, 5, 10))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
