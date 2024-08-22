# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Conformer',
        arch='small',
        patch_size=32,
        drop_path_rate=0.1,
        init_cfg=None),
    neck=None,
    head=dict(
        type='ConformerHead',
        num_classes=1000,
        in_channels=[1024, 384],
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
