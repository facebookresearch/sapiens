# data settings

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(384, 384),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
]

val_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type='NoCaps',
        data_root='data/nocaps/',
        data_prefix=dict(img_path='images/'),
        ann_file='annotations/nocaps_val_4500_captions.json',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(
    type='NocapsSave',
    save_dir='./',
)

# # If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
