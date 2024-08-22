# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.3, momentum=0.9, weight_decay=1e-6))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
val_cfg = dict()
test_cfg = dict()
