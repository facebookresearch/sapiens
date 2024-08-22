# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from mmengine.utils import digit_version

from mmdet.registry import MODELS

if digit_version(torch.__version__) >= digit_version('1.7.0'):
    from torch.nn import SiLU
else:

    class SiLU(nn.Module):
        """Sigmoid Weighted Liner Unit."""

        def __init__(self, inplace=True):
            super().__init__()

        def forward(self, inputs) -> torch.Tensor:
            return inputs * torch.sigmoid(inputs)


MODELS.register_module(module=SiLU, name='SiLU')
