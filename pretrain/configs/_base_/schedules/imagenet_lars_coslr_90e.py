# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=1.6, momentum=0.9, weight_decay=0.))

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
val_cfg = dict()
test_cfg = dict()
