# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        drop_path_rate=0.2,
        layer_scale_init_value=0.,
        use_grn=True,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.2),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]),
)
