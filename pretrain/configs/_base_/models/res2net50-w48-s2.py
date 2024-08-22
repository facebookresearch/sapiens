model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=2,
        base_width=48,
        deep_stem=False,
        avg_down=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
