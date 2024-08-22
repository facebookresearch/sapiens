# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0),
)

# learning policy
param_scheduler = [
    dict(type='ConstantLR', factor=0.1, by_epoch=False, begin=0, end=5000),
    dict(type='PolyLR', eta_min=0, by_epoch=False, begin=5000)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)
