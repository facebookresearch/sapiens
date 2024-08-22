# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch

try:
    import torch_npu  # noqa: F401
    import torch_npu.npu.utils as npu_utils

    # Enable operator support for dynamic shape and
    # binary operator support on the NPU.
    npu_jit_compile = bool(os.getenv('NPUJITCompile', False))
    torch.npu.set_compile_mode(jit_compile=npu_jit_compile)
    IS_NPU_AVAILABLE = hasattr(torch, 'npu') and torch.npu.is_available()
except Exception:
    IS_NPU_AVAILABLE = False

try:
    import torch_dipu  # noqa: F401
    IS_DIPU_AVAILABLE = True
except Exception:
    IS_DIPU_AVAILABLE = False


def get_max_cuda_memory(device: Optional[torch.device] = None) -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.cuda.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    torch.cuda.reset_peak_memory_stats()
    return int(mem_mb.item())


def is_cuda_available() -> bool:
    """Returns True if cuda devices exist."""
    return torch.cuda.is_available()


def is_npu_available() -> bool:
    """Returns True if Ascend PyTorch and npu devices exist."""
    return IS_NPU_AVAILABLE


def is_mlu_available() -> bool:
    """Returns True if Cambricon PyTorch and mlu devices exist."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def is_mps_available() -> bool:
    """Return True if mps devices exist.

    It's specialized for mac m1 chips and require torch version 1.12 or higher.
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def is_dipu_available() -> bool:
    return IS_DIPU_AVAILABLE


def is_npu_support_full_precision() -> bool:
    """Returns True if npu devices support full precision training."""
    version_of_support_full_precision = 220
    return IS_NPU_AVAILABLE and npu_utils.get_soc_version(
    ) >= version_of_support_full_precision


DEVICE = 'cpu'
if is_npu_available():
    DEVICE = 'npu'
elif is_cuda_available():
    DEVICE = 'cuda'
elif is_mlu_available():
    DEVICE = 'mlu'
elif is_mps_available():
    DEVICE = 'mps'
elif is_dipu_available():
    DEVICE = 'dipu'


def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: cuda | npu | mlu | mps | cpu.
    """
    return DEVICE
