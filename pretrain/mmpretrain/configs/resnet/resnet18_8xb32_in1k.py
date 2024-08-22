# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_bs32 import *
    from .._base_.default_runtime import *
    from .._base_.models.resnet18 import *
    from .._base_.schedules.imagenet_bs256 import *
