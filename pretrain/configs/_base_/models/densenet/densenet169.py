# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='169'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1664,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
