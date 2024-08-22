model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepVGG',
        arch='B3',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2560,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            mode='classy_vision',
            num_classes=1000),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)
