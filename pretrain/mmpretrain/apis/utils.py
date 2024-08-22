# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from mmpretrain.utils import require


@require('torch>=1.9.0', 'https://pytorch.org/get-started/locally/')
@require('accelerate')
def dispatch_model(
    model,
    device_map: Union[str, dict],
    max_memory: Optional[dict] = None,
    no_split_module_classes: Optional[List[str]] = None,
    offload_folder: str = None,
    offload_buffers: bool = False,
    preload_module_classes: Optional[List[str]] = None,
):
    """Split and dispatch a model across devices.

    The function depends on the `accelerate` package. Refers to
    https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling

    Args:
        model (torch.nn.Module): The model to dispatch.
        device_map (str | dict | None): A map that specifies where each
            submodule should go. It doesn't need to be refined to each
            parameter/buffer name, once a given module name is inside, every
            submodule of it will be sent to the same device. You can use
            `device_map="auto"` to automatically generate the device map.
            Defaults to None.
        max_memory (dict | None): A dictionary device identifier to maximum
            memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset. Defaults to None.
        no_split_module_classes (List[str] | None): A list of layer class names
            that should never be split across device (for instance any layer
            that has a residual connection). If None, try to get the settings
            from the model class. Defaults to None.
        offload_folder (str | None): If the `device_map` contains any value
            `"disk"`, the folder where we will offload weights.
        offload_buffers (bool): In the layers that are offloaded on the CPU
            or the hard drive, whether or not to offload the buffers as
            well as the parameters. Defaults to False.
        preload_module_classes (List[str] | None): A list of classes whose
            instances should load all their weights (even in the submodules) at
            the beginning of the forward. This should only be used for classes
            that have submodules which are registered but not called directly
            during the forward, for instance if a `dense` linear layer is
            registered, but at forward, `dense.weight` and `dense.bias` are
            used in some operations instead of calling `dense` directly.
            Defaults to None.
    """
    from accelerate import dispatch_model, infer_auto_device_map

    # Check valid device_map string.
    valid_map_option = ['auto', 'balanced', 'balanced_low_0', 'sequential']
    if isinstance(device_map, str) and device_map not in valid_map_option:
        raise ValueError('If passing a string for `device_map`, please choose '
                         f'from {valid_map_option}.')

    # Generate device map automatically
    if isinstance(device_map, str):
        if no_split_module_classes is None:
            no_split_module_classes = getattr(model, '_no_split_modules', None)
        if no_split_module_classes is None:
            raise ValueError(f'{model.__class__.__name__} does not support '
                             f"`device_map='{device_map}'` yet.")

        if device_map != 'sequential':
            from accelerate.utils import get_balanced_memory
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=None,
                low_zero=(device_map == 'balanced_low_0'),
            )
            max_memory[0] *= 0.9
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
            dtype=None,
        )

    if 'disk' in device_map.values():
        if offload_folder is None:
            raise ValueError(
                'The current `device_map` had weights offloaded to the disk. '
                'Please provide an `offload_folder` for them.')
        os.makedirs(offload_folder, exist_ok=True)

    main_device = next(
        (d for d in device_map.values() if d not in ['cpu', 'disk']), 'cpu')

    model = dispatch_model(
        model,
        device_map=device_map,
        main_device=main_device,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        preload_module_classes=preload_module_classes,
    )
    if hasattr(model, 'data_preprocessor'):
        model.data_preprocessor._device = torch.device(main_device)
    return model


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """A context manager under which models are initialized with all parameters
    on the meta device.

    With this context manager, we can create an empty model. Useful when just
    initializing the model would blow the available RAM.

    Besides move the parameters to meta device, this method will also avoid
    load checkpoint from `mmengine.runner.load_checkpoint` and
    `transformers.PreTrainedModel.from_pretrained`.

    Modified from https://github.com/huggingface/accelerate

    Args:
        include_buffers (bool): Whether put all buffers on the meta device
            during initialization.
    """
    device = torch.device('meta')

    # move parameter and buffer to meta device
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer
        # See https://github.com/huggingface/accelerate/pull/699
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ['empty', 'zeros', 'ones', 'full']
        }

    def register_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs)

    def register_buffer(module, name, buffer, *args, **kwargs):
        old_register_buffer(module, name, buffer, *args, **kwargs)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):

        def wrapper(*args, **kwargs):
            kwargs['device'] = device
            return fn(*args, **kwargs)

        return wrapper

    # Patch load_checkpoint
    import mmengine.runner.checkpoint as mmengine_load
    old_load_checkpoint = mmengine_load.load_checkpoint

    def patch_load_checkpoint(*args, **kwargs):
        return {}

    # Patch transformers from pretrained
    try:
        from transformers import PreTrainedModel
        from transformers.models.auto.auto_factory import (AutoConfig,
                                                           _BaseAutoModelClass)
        with_transformers = True
    except ImportError:
        with_transformers = False

    @classmethod
    def patch_auto_model(cls, pretrained_model_name_or_path, *model_args,
                         **kwargs):
        cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                         *model_args, **kwargs)
        return cls.from_config(cfg)

    @classmethod
    def patch_pretrained_model(cls, pretrained_model_name_or_path, *model_args,
                               **kwargs):
        cfg = cls.config_class.from_pretrained(pretrained_model_name_or_path,
                                               *model_args, **kwargs)
        return cls(cfg)

    if with_transformers:
        old_pretrained_model = PreTrainedModel.from_pretrained
        old_auto_model = _BaseAutoModelClass.from_pretrained

    try:
        nn.Module.register_parameter = register_parameter
        mmengine_load.load_checkpoint = patch_load_checkpoint
        if with_transformers:
            PreTrainedModel.from_pretrained = patch_pretrained_model
            _BaseAutoModelClass.from_pretrained = patch_auto_model
        if include_buffers:
            nn.Module.register_buffer = register_buffer
            for func in tensor_constructors_to_patch.keys():
                tensor_constructor = patch_tensor_constructor(
                    getattr(torch, func))
                setattr(torch, func, tensor_constructor)
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        mmengine_load.load_checkpoint = old_load_checkpoint
        if with_transformers:
            PreTrainedModel.from_pretrained = old_pretrained_model
            _BaseAutoModelClass.from_pretrained = old_auto_model
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
            for func, ori in tensor_constructors_to_patch.items():
                setattr(torch, func, ori)


def compute_module_sizes(
        model: nn.Module,
        dtype: Union[str, torch.dtype, None] = None,
        special_dtypes: Optional[Dict[str, Union[str, torch.dtype]]] = None):
    """Compute the size of each submodule of a given model."""

    def get_dtype(dtype):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None:
            assert issubclass(dtype, torch.dtype)
        return dtype

    def dtype_bytes(dtype: torch.dtype):
        if dtype is torch.bool:
            return 1
        if dtype.is_floating_point:
            return torch.finfo(dtype).bits / 8
        else:
            return torch.iinfo(dtype).bits / 8

    if dtype is not None:
        dtype = get_dtype(dtype)
        dtype_size = dtype_bytes(dtype)

    if special_dtypes is not None:
        special_dtypes = {
            key: dtype_bytes(dtype)
            for key, dtype in special_dtypes.items()
        }

    module_sizes = defaultdict(int)
    for name, tensor in chain(
            model.named_parameters(recurse=True),
            model.named_buffers(recurse=True)):
        if special_dtypes is not None and name in special_dtypes:
            size = tensor.numel() * special_dtypes[name]
        elif dtype is None:
            size = tensor.numel() * tensor.element_size()
        else:
            size = tensor.numel() * min(dtype_size, tensor.element_size())
        name_parts = name.split('.')
        for idx in range(len(name_parts) + 1):
            module_sizes['.'.join(name_parts[:idx])] += size

    return module_sizes
