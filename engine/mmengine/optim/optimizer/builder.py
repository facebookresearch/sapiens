# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
from typing import List, Union

import torch
import torch.nn as nn

from mmengine.config import Config, ConfigDict
from mmengine.device import is_npu_available, is_npu_support_full_precision
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS
from .optimizer_wrapper import OptimWrapper


def register_torch_optimizers() -> List[str]:
    """Register optimizers in ``torch.optim`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def register_torch_npu_optimizers() -> List[str]:
    """Register optimizers in ``torch npu`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    if not is_npu_available():
        return []

    import torch_npu
    if not hasattr(torch_npu, 'optim'):
        return []

    torch_npu_optimizers = []
    for module_name in dir(torch_npu.optim):
        if module_name.startswith('__') or module_name in OPTIMIZERS:
            continue
        _optim = getattr(torch_npu.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_npu_optimizers.append(module_name)
    return torch_npu_optimizers


NPU_OPTIMIZERS = register_torch_npu_optimizers()


def register_dadaptation_optimizers() -> List[str]:
    """Register optimizers in ``dadaptation`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    dadaptation_optimizers = []
    try:
        import dadaptation
    except ImportError:
        pass
    else:
        for module_name in ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD']:
            _optim = getattr(dadaptation, module_name)
            if inspect.isclass(_optim) and issubclass(_optim,
                                                      torch.optim.Optimizer):
                OPTIMIZERS.register_module(module=_optim)
                dadaptation_optimizers.append(module_name)
    return dadaptation_optimizers


DADAPTATION_OPTIMIZERS = register_dadaptation_optimizers()


def register_lion_optimizers() -> List[str]:
    """Register Lion optimizer to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    optimizers = []
    try:
        from lion_pytorch import Lion
    except ImportError:
        pass
    else:
        OPTIMIZERS.register_module(module=Lion)
        optimizers.append('Lion')
    return optimizers


LION_OPTIMIZERS = register_lion_optimizers()


def register_sophia_optimizers() -> List[str]:
    """Register Sophia optimizer to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    optimizers = []
    try:
        import Sophia
    except ImportError:
        pass
    else:
        for module_name in dir(Sophia):
            _optim = getattr(Sophia, module_name)
            if inspect.isclass(_optim) and issubclass(_optim,
                                                      torch.optim.Optimizer):
                OPTIMIZERS.register_module(module=_optim)
                optimizers.append(module_name)
    return optimizers


SOPHIA_OPTIMIZERS = register_sophia_optimizers()


def register_bitsandbytes_optimizers() -> List[str]:
    """Register optimizers in ``bitsandbytes`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    dadaptation_optimizers = []
    try:
        import bitsandbytes as bnb
    except ImportError:
        pass
    else:
        for module_name in [
                'AdamW8bit', 'Adam8bit', 'Adagrad8bit', 'PagedAdam8bit',
                'PagedAdamW8bit', 'LAMB8bit', 'LARS8bit', 'RMSprop8bit',
                'Lion8bit', 'PagedLion8bit', 'SGD8bit'
        ]:
            _optim = getattr(bnb.optim, module_name)
            if inspect.isclass(_optim) and issubclass(_optim,
                                                      torch.optim.Optimizer):
                OPTIMIZERS.register_module(module=_optim)
                dadaptation_optimizers.append(module_name)
    return dadaptation_optimizers


BITSANDBYTES_OPTIMIZERS = register_bitsandbytes_optimizers()


def register_transformers_optimizers():
    transformer_optimizers = []
    try:
        from transformers import Adafactor
    except ImportError:
        pass
    else:
        OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)
        transformer_optimizers.append('Adafactor')
    return transformer_optimizers


TRANSFORMERS_OPTIMIZERS = register_transformers_optimizers()


def build_optim_wrapper(model: nn.Module,
                        cfg: Union[dict, Config, ConfigDict]) -> OptimWrapper:
    """Build function of OptimWrapper.

    If ``constructor`` is set in the ``cfg``, this method will build an
    optimizer wrapper constructor, and use optimizer wrapper constructor to
    build the optimizer wrapper. If ``constructor`` is not set, the
    ``DefaultOptimWrapperConstructor`` will be used by default.

    Args:
        model (nn.Module): Model to be optimized.
        cfg (dict): Config of optimizer wrapper, optimizer constructor and
            optimizer.

    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    optim_wrapper_cfg = copy.deepcopy(cfg)
    constructor_type = optim_wrapper_cfg.pop('constructor',
                                             'DefaultOptimWrapperConstructor')
    paramwise_cfg = optim_wrapper_cfg.pop('paramwise_cfg', None)

    # Since the current generation of NPU(Ascend 910) only supports
    # mixed precision training, here we turn on mixed precision
    # to make the training normal
    if is_npu_available() and not is_npu_support_full_precision():
        optim_wrapper_cfg['type'] = 'AmpOptimWrapper'

    optim_wrapper_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
        dict(
            type=constructor_type,
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg))
    optim_wrapper = optim_wrapper_constructor(model)
    return optim_wrapper
