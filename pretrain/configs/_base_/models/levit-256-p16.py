# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='LeViT',
        arch='256',
        img_size=224,
        patch_size=16,
        drop_path_rate=0,
        attn_ratio=2,
        mlp_ratio=2,
        out_indices=(2, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LeViTClsHead',
        num_classes=1000,
        in_channels=512,
        distillation=True,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0),
    ]))
