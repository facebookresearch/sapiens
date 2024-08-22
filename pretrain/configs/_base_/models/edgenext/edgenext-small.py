# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EdgeNeXt',
        arch='small',
        out_indices=(3, ),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=304,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
