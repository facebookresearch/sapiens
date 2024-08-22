# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mmcv
import mmengine
from mmengine.utils import digit_version

from .apis import *  # noqa: F401, F403
from .version import __version__

mmcv_minimum_version = '2.0.0'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.8.3'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

__all__ = ['__version__']
