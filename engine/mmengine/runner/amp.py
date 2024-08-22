# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import contextmanager
from typing import Optional

import torch

from mmengine.device import (get_device, is_cuda_available, is_mlu_available,
                             is_npu_available)
from mmengine.logging import print_log
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


@contextmanager
def autocast(device_type: Optional[str] = None,
             dtype: Optional[torch.dtype] = None,
             enabled: bool = True,
             cache_enabled: Optional[bool] = None):
    """A wrapper of ``torch.autocast`` and ``toch.cuda.amp.autocast``.

    Pytorch 1.5.0 provide ``torch.cuda.amp.autocast`` for running in
    mixed precision , and update it to ``torch.autocast`` in 1.10.0.
    Both interfaces have different arguments, and ``torch.autocast``
    support running with cpu additionally.

    This function provides a unified interface by wrapping
    ``torch.autocast`` and ``torch.cuda.amp.autocast``, which resolves the
    compatibility issues that ``torch.cuda.amp.autocast`` does not support
    running mixed precision with cpu, and both contexts have different
    arguments. We suggest users using this function in the code
    to achieve maximized compatibility of different PyTorch versions.

    Note:
        ``autocast`` requires pytorch version >= 1.5.0. If pytorch version
        <= 1.10.0 and cuda is not available, it will raise an error with
        ``enabled=True``, since ``torch.cuda.amp.autocast`` only support cuda
        mode.

    Examples:
         >>> # case1: 1.10 > Pytorch version >= 1.5.0
         >>> with autocast():
         >>>    # run in mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu')::
         >>>    # raise error, torch.cuda.amp.autocast only support cuda mode.
         >>>    pass
         >>> # case2: Pytorch version >= 1.10.0
         >>> with autocast():
         >>>    # default cuda mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu'):
         >>>    # cpu mixed precision context
         >>>    pass
         >>> with autocast(
         >>>     device_type='cuda', enabled=True, cache_enabled=True):
         >>>    # enable precision context with more specific arguments.
         >>>    pass

    Args:
        device_type (str, required):  Whether to use 'cuda' or 'cpu' device.
        enabled(bool):  Whether autocasting should be enabled in the region.
            Defaults to True
        dtype (torch_dtype, optional):  Whether to use ``torch.float16`` or
            ``torch.bfloat16``.
        cache_enabled(bool, optional):  Whether the weight cache inside
            autocast should be enabled.
    """
    # If `enabled` is True, enable an empty context and all calculations
    # are performed under fp32.
    assert digit_version(TORCH_VERSION) >= digit_version('1.5.0'), (
        'The minimum pytorch version requirements of mmengine is 1.5.0, but '
        f'got {TORCH_VERSION}')

    if (digit_version('1.5.0') <= digit_version(TORCH_VERSION) <
            digit_version('1.10.0')):
        # If pytorch version is between 1.5.0 and 1.10.0, the default value of
        # dtype for `torch.cuda.amp.autocast` is torch.float16.
        assert (
            device_type == 'cuda' or device_type == 'mlu'
            or device_type is None), (
                'Pytorch version under 1.10.0 only supports running automatic '
                'mixed training with cuda or mlu')
        if dtype is not None or cache_enabled is not None:
            print_log(
                f'{dtype} and {device_type} will not work for '
                '`autocast` since your Pytorch version: '
                f'{TORCH_VERSION} <= 1.10.0',
                logger='current',
                level=logging.WARNING)

        if is_npu_available():
            with torch.npu.amp.autocast(enabled=enabled):
                yield
        elif is_mlu_available():
            with torch.mlu.amp.autocast(enabled=enabled):
                yield
        elif is_cuda_available():
            with torch.cuda.amp.autocast(enabled=enabled):
                yield
        else:
            if not enabled:
                yield
            else:
                raise RuntimeError(
                    'If pytorch versions is between 1.5.0 and 1.10, '
                    '`autocast` is only available in gpu mode')

    else:
        # Modified from https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py # noqa: E501
        # This code should update with the `torch.autocast`.
        if cache_enabled is None:
            cache_enabled = torch.is_autocast_cache_enabled()
        device = get_device()
        device_type = device if device_type is None else device_type

        if device_type == 'cuda':
            if dtype is None:
                dtype = torch.get_autocast_gpu_dtype()

            if dtype == torch.bfloat16 and not \
                    torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    'Current CUDA Device does not support bfloat16. Please '
                    'switch dtype to float16.')

        elif device_type == 'cpu':
            if dtype is None:
                dtype = torch.bfloat16
            assert dtype == torch.bfloat16, (
                'In CPU autocast, only support `torch.bfloat16` dtype')

        elif device_type == 'mlu':
            pass

        elif device_type == 'npu':
            pass

        else:
            # Device like MPS does not support fp16 training or testing.
            # If an inappropriate device is set and fp16 is enabled, an error
            # will be thrown.
            if enabled is False:
                yield
                return
            else:
                raise ValueError('User specified autocast device_type must be '
                                 f'cuda or cpu, but got {device_type}')

        with torch.autocast(
                device_type=device_type,
                enabled=enabled,
                dtype=dtype,
                cache_enabled=cache_enabled):
            yield
