# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from .base import BaseStrategy
from .colossalai import ColossalAIStrategy
from .deepspeed import DeepSpeedStrategy
from .distributed import DDPStrategy
from .single_device import SingleDeviceStrategy

__all__ = [
    'BaseStrategy', 'DDPStrategy', 'SingleDeviceStrategy', 'DeepSpeedStrategy',
    'ColossalAIStrategy'
]

if digit_version(TORCH_VERSION) >= digit_version('2.0.0'):
    try:
        from .fsdp import FSDPStrategy  # noqa:F401
        __all__.append('FSDPStrategy')
    except:  # noqa: E722
        pass
