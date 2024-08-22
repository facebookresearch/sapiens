# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=200, by_epoch=True, begin=0, end=200)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
