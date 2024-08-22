# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w48'),
    neck=[
        dict(type='HRFuseScales', in_channels=(48, 96, 192, 384)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=1000,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
