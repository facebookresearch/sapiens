# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.25, by_epoch=False, begin=0, end=2500),
    dict(
        type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
