# dataset settings
## to create the sample_ids from tsv.
# cut -f1 /uca_transient_a/exported_sample_tsv/codec_avatar_shutterstock_image_editorial_model_v3_no_user_data.tsv > shutterstock_sample_ids.txt
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(
        type='RandomResizedCrop',
        scale=1024,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=512,
    num_workers=8,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=True), ## use when indexing
    sampler=None, ## use when iterable
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='Shutterstock',
        data_root='data/shutterstock',
        airstore_id='codec_avatar_shutterstock_image_editorial_model_v3_no_user_data',
        split='train',
        pipeline=train_pipeline))
