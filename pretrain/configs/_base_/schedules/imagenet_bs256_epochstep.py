# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004))

# learning policy
param_scheduler = dict(type='StepLR', by_epoch=True, step_size=1, gamma=0.98)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
