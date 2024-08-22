# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.device import (is_cuda_available, is_mlu_available,
                             is_mps_available, is_npu_available)

IS_MLU_AVAILABLE = is_mlu_available()
IS_MPS_AVAILABLE = is_mps_available()
IS_CUDA_AVAILABLE = is_cuda_available()
IS_NPU_AVAILABLE = is_npu_available()
