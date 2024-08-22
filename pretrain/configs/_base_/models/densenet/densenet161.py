# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='161'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2208,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
