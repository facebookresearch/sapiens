# model settings
model = dict(
    type='ImageClassifier',
    # `es` means EfficientNet-EdgeTPU-S arch
    backbone=dict(type='EfficientNet', arch='es', act_cfg=dict(type='ReLU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
