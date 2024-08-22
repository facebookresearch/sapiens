# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='nano',
        drop_path_rate=0.1,
        layer_scale_init_value=0.,
        use_grn=True,
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=640,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.2),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
)
