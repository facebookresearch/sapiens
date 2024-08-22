model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormer',
        arch='l1',
        drop_path_rate=0,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='EfficientFormerClsHead', in_channels=448, num_classes=1000))
