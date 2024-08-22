# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='161'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2208,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
