# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing
from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union

import torch.nn as nn
from rich import box
from rich.console import Console
from rich.table import Table
from torch import Tensor

from .jit_analysis import JitModelAnalysis
from .jit_handles import (Handle, addmm_flop_jit, batchnorm_flop_jit,
                          bmm_flop_jit, conv_flop_jit, einsum_flop_jit,
                          elementwise_flop_counter, generic_activation_jit,
                          linear_flop_jit, matmul_flop_jit, norm_flop_counter)

# A dictionary that maps supported operations to their flop count jit handles.
_DEFAULT_SUPPORTED_FLOP_OPS: Dict[str, Handle] = {
    'aten::addmm': addmm_flop_jit,
    'aten::bmm': bmm_flop_jit,
    'aten::_convolution': conv_flop_jit,
    'aten::einsum': einsum_flop_jit,
    'aten::matmul': matmul_flop_jit,
    'aten::mm': matmul_flop_jit,
    'aten::linear': linear_flop_jit,
    # You might want to ignore BN flops due to inference-time fusion.
    # Use `set_op_handle("aten::batch_norm", None)
    'aten::batch_norm': batchnorm_flop_jit,
    'aten::group_norm': norm_flop_counter(2),
    'aten::layer_norm': norm_flop_counter(2),
    'aten::instance_norm': norm_flop_counter(1),
    'aten::upsample_nearest2d': elementwise_flop_counter(0, 1),
    'aten::upsample_bilinear2d': elementwise_flop_counter(0, 4),
    'aten::adaptive_avg_pool2d': elementwise_flop_counter(1, 0),
    'aten::grid_sampler': elementwise_flop_counter(0, 4),  # assume bilinear
}

# A dictionary that maps supported operations to
# their activation count handles.
_DEFAULT_SUPPORTED_ACT_OPS: Dict[str, Handle] = {
    'aten::_convolution': generic_activation_jit('conv'),
    'aten::addmm': generic_activation_jit(),
    'aten::bmm': generic_activation_jit(),
    'aten::einsum': generic_activation_jit(),
    'aten::matmul': generic_activation_jit(),
    'aten::linear': generic_activation_jit(),
}


class FlopAnalyzer(JitModelAnalysis):
    """Provides access to per-submodule model flop count obtained by tracing a
    model with pytorch's jit tracing functionality.

    By default, comes with standard flop counters for a few common operators.

    Note:
        - Flop is not a well-defined concept. We just produce our best
          estimate.
        - We count one fused multiply-add as one flop.

    Handles for additional operators may be added, or the default ones
    overwritten, using the ``.set_op_handle(name, func)`` method.
    See the method documentation for details.
    Flop counts can be obtained as:

    - ``.total(module_name="")``: total flop count for the module
    - ``.by_operator(module_name="")``: flop counts for the module, as a
      Counter over different operator types
    - ``.by_module()``: Counter of flop counts for all submodules
    - ``.by_module_and_operator()``: dictionary indexed by descendant of
      Counters over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Modified from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py

    Args:
        model (nn.Module): The model to analyze.
        inputs (Union[Tensor, Tuple[Tensor, ...]]): The input to the model.

    Examples:
        >>> import torch.nn as nn
        >>> import torch
        >>> class TestModel(nn.Module):
        ...    def __init__(self):
        ...        super().__init__()
        ...        self.fc = nn.Linear(in_features=1000, out_features=10)
        ...        self.conv = nn.Conv2d(
        ...            in_channels=3, out_channels=10, kernel_size=1
        ...        )
        ...        self.act = nn.ReLU()
        ...    def forward(self, x):
        ...        return self.fc(self.act(self.conv(x)).flatten(1))
        >>> model = TestModel()
        >>> inputs = (torch.randn((1,3,10,10)),)
        >>> flops = FlopAnalyzer(model, inputs)
        >>> flops.total()
        13000
        >>> flops.total("fc")
        10000
        >>> flops.by_operator()
        Counter({"addmm" : 10000, "conv" : 3000})
        >>> flops.by_module()
        Counter({"" : 13000, "fc" : 10000, "conv" : 3000, "act" : 0})
        >>> flops.by_module_and_operator()
        {"" : Counter({"addmm" : 10000, "conv" : 3000}),
        "fc" : Counter({"addmm" : 10000}),
        "conv" : Counter({"conv" : 3000}),
        "act" : Counter()
        }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_FLOP_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


class ActivationAnalyzer(JitModelAnalysis):
    """Provides access to per-submodule model activation count obtained by
    tracing a model with pytorch's jit tracing functionality.

    By default, comes with standard activation counters for convolutional and
    dot-product operators. Handles for additional operators may be added, or
    the default ones overwritten, using the ``.set_op_handle(name, func)``
    method. See the method documentation for details. Activation counts can be
    obtained as:

    - ``.total(module_name="")``: total activation count for a module
    - ``.by_operator(module_name="")``: activation counts for the module,
      as a Counter over different operator types
    - ``.by_module()``: Counter of activation counts for all submodules
    - ``.by_module_and_operator()``: dictionary indexed by descendant of
      Counters over different operator types

    An operator is treated as within a module if it is executed inside the
    module's ``__call__`` method. Note that this does not include calls to
    other methods of the module or explicit calls to ``module.forward(...)``.

    Modified from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/activation_count.py

    Args:
        model (nn.Module): The model to analyze.
        inputs (Union[Tensor, Tuple[Tensor, ...]]): The input to the model.

    Examples:
        >>> import torch.nn as nn
        >>> import torch
        >>> class TestModel(nn.Module):
        ...     def __init__(self):
        ...        super().__init__()
        ...        self.fc = nn.Linear(in_features=1000, out_features=10)
        ...        self.conv = nn.Conv2d(
        ...            in_channels=3, out_channels=10, kernel_size=1
        ...        )
        ...        self.act = nn.ReLU()
        ...    def forward(self, x):
        ...        return self.fc(self.act(self.conv(x)).flatten(1))
        >>> model = TestModel()
        >>> inputs = (torch.randn((1,3,10,10)),)
        >>> acts = ActivationAnalyzer(model, inputs)
        >>> acts.total()
        1010
        >>> acts.total("fc")
        10
        >>> acts.by_operator()
        Counter({"conv" : 1000, "addmm" : 10})
        >>> acts.by_module()
        Counter({"" : 1010, "fc" : 10, "conv" : 1000, "act" : 0})
        >>> acts.by_module_and_operator()
        {"" : Counter({"conv" : 1000, "addmm" : 10}),
        "fc" : Counter({"addmm" : 10}),
        "conv" : Counter({"conv" : 1000}),
        "act" : Counter()
        }
    """

    def __init__(
        self,
        model: nn.Module,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        super().__init__(model=model, inputs=inputs)
        self.set_op_handle(**_DEFAULT_SUPPORTED_ACT_OPS)

    __init__.__doc__ = JitModelAnalysis.__init__.__doc__


def flop_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """Given a model and an input to the model, compute the per-operator Gflops
    of the given model.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and
            the value is a function that takes (inputs, outputs) of the op.
            We count one Multiply-Add as one FLOP.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
        gflops for each operation and a Counter that records the number of
        unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    flop_counter = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
    giga_flops = defaultdict(float)
    for op, flop in flop_counter.by_operator().items():
        giga_flops[op] = flop / 1e9
    return giga_flops, flop_counter.unsupported_ops()


def activation_count(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    supported_ops: Optional[Dict[str, Handle]] = None,
) -> Tuple[DefaultDict[str, float], Counter[str]]:
    """Given a model and an input to the model, compute the total number of
    activations of the model.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/activation_count.py

    Args:
        model (nn.Module): The model to compute activation counts.
        inputs (tuple): Inputs that are passed to `model` to count activations.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
        activation (mega) for each operation and a Counter that records the
        number of unsupported operations.
    """
    if supported_ops is None:
        supported_ops = {}
    act_counter = ActivationAnalyzer(model,
                                     inputs).set_op_handle(**supported_ops)
    mega_acts = defaultdict(float)
    for op, act in act_counter.by_operator().items():
        mega_acts[op] = act / 1e6
    return mega_acts, act_counter.unsupported_ops()


def parameter_count(model: nn.Module) -> typing.DefaultDict[str, int]:
    """Count parameters of a model and its submodules.

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/parameter_count.py

    Args:
        model (nn.Module): the model to count parameters.

    Returns:
        dict[str, int]: the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    count = defaultdict(int)  # type: typing.DefaultDict[str, int]
    for name, param in model.named_parameters():
        size = param.numel()
        name = name.split('.')
        for k in range(0, len(name) + 1):
            prefix = '.'.join(name[:k])
            count[prefix] += size
    return count


def parameter_count_table(model: nn.Module, max_depth: int = 3) -> str:
    """Format the parameter count of the model (and its submodules or
    parameters)

    Adopted from
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/parameter_count.py

    Args:
        model (nn.Module): the model to count parameters.
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameter_count(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape)
        for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    rows: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e8:
            return f'{x / 1e9:.1f}G'
        if x > 1e5:
            return f'{x / 1e6:.1f}M'
        if x > 1e2:
            return f'{x / 1e3:.1f}K'
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count('.') == lvl and name.startswith(prefix):
                indent = ' ' * (lvl + 1)
                if name in param_shape:
                    rows.append(
                        (indent + name, indent + str(param_shape[name])))
                else:
                    rows.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + '.')

    rows.append(('model', format_size(count.pop(''))))
    fill(0, '')

    table = Table(
        title=f'parameter count of {model.__class__.__name__}', box=box.ASCII2)
    table.add_column('name')
    table.add_column('#elements or shape')

    for row in rows:
        table.add_row(*row)

    console = Console()
    with console.capture() as capture:
        console.print(table, end='')

    return capture.get()
