# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvMixer', arch='768/32', act_cfg=dict(type='ReLU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
