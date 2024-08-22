# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w40'),
    neck=[
        dict(type='HRFuseScales', in_channels=(40, 80, 160, 320)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=1000,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
