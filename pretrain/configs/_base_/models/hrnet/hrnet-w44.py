# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w44'),
    neck=[
        dict(type='HRFuseScales', in_channels=(44, 88, 176, 352)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=1000,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
