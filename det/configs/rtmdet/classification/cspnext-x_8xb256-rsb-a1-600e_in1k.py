# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'

model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    head=dict(in_channels=1280))
