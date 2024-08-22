# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ShuffleNetV1', groups=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=960,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
