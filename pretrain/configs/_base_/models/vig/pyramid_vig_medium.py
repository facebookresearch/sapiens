# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PyramidVig',
        arch='medium',
        k=9,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN'),
        graph_conv_type='mr',
        graph_conv_bias=True,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0.1,
        norm_eval=False,
        frozen_stages=0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='VigClsHead',
        num_classes=1000,
        in_channels=768,
        hidden_dim=1024,
        act_cfg=dict(type='GELU'),
        dropout=0.,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
