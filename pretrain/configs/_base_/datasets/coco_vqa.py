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
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(480, 480),
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='CleanCaption',
        keys=['question'],
    ),
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    ),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='COCOVQA',
        data_root='data/coco',
        data_prefix='train2014',
        question_file=
        'annotations/v2_OpenEnded_mscoco_train2014_questions.json',
        ann_file='annotations/v2_mscoco_train2014_annotations.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='COCOVQA',
        data_root='data/coco',
        data_prefix='val2014',
        question_file='annotations/v2_OpenEnded_mscoco_val2014_questions.json',
        ann_file='annotations/v2_mscoco_val2014_annotations.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='VQAAcc')

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='COCOVQA',
        data_root='data/coco',
        data_prefix='test2015',
        question_file=  # noqa: E251
        'annotations/v2_OpenEnded_mscoco_test2015_questions.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='ReportVQA', file_path='vqa_test.json')
