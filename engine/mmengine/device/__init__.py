# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .utils import (get_device, get_max_cuda_memory, is_cuda_available,
                    is_dipu_available, is_mlu_available, is_mps_available,
                    is_npu_available, is_npu_support_full_precision)

__all__ = [
    'get_max_cuda_memory', 'get_device', 'is_cuda_available',
    'is_mlu_available', 'is_mps_available', 'is_npu_available',
    'is_dipu_available', 'is_npu_support_full_precision'
]
