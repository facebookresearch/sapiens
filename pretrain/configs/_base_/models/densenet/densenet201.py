# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='201'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
