# data settings
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
        scale=(480, 480),
        crop_ratio_range=(0.5, 1.0),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='simple_increasing',  # slightly different from LAVIS
        num_policies=2,
        magnitude_level=5),
    dict(type='CleanCaption', keys=['question', 'gt_answer']),
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(480, 480),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys=['question']),
    dict(
        type='PackInputs',
        algorithm_keys=['question'],
        meta_keys=['question_id']),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # VQAv2 train
            dict(
                type='COCOVQA',
                data_root='data/coco',
                data_prefix='train2014',
                question_file=
                'annotations/v2_OpenEnded_mscoco_train2014_questions.json',
                ann_file='annotations/v2_mscoco_train2014_annotations.json',
                pipeline=train_pipeline,
            ),
            # VQAv2 val
            dict(
                type='COCOVQA',
                data_root='data/coco',
                data_prefix='val2014',
                question_file=
                'annotations/v2_OpenEnded_mscoco_val2014_questions.json',
                ann_file='annotations/v2_mscoco_val2014_annotations.json',
                pipeline=train_pipeline,
            ),
            # Visual Genome
            dict(
                type='VisualGenomeQA',
                data_root='visual_genome',
                data_prefix='image',
                ann_file='question_answers.json',
                pipeline=train_pipeline,
            )
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='COCOVQA',
        data_root='data/coco',
        data_prefix='test2015',
        question_file=
        'annotations/v2_OpenEnded_mscoco_test2015_questions.json',  # noqa: E501
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='ReportVQA', file_path='vqa_test.json')
