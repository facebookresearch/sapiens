# model settings
# Only for evaluation
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerV2',
        arch='large',
        img_size=256,
        drop_path_rate=0.2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
