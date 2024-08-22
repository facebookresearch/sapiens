# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PCPVT',
        arch='base',
        in_channels=3,
        out_indices=(3, ),
        qkv_bias=True,
        norm_cfg=dict(type='LN', eps=1e-06),
        norm_after_stage=[False, False, False, True],
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
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
