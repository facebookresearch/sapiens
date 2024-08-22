# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/jit_handles.py

import typing
from collections import Counter, OrderedDict
from typing import Any, Callable, List, Optional, Union

import numpy as np

try:
    from math import prod  # type: ignore
except ImportError:
    from numpy import prod as _prod  # type: ignore

    # Patch `numpy.prod` to avoid overflow on Windows by converting its result
    # from `np.int32` to `int`.
    def prod(*args, **kwargs):  # type: ignore
        return _prod(*args, **kwargs).item()


Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], int]]


def get_shape(val: Any) -> Optional[List[int]]:
    """Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None  # type: ignore


"""
Below are flop/activation counters for various ops.
Every counter has the following signature:

Args:
    inputs (list(torch._C.Value)):
        The inputs of the op in the form of a list of jit object.
    outputs (list(torch._C.Value)):
        The outputs of the op in the form of a list of jit object.

Returns:
    number: The number of flops/activations for the operation.
    or Counter[str]
"""


def generic_activation_jit(op_name: Optional[str] = None) -> Handle:
    """This method returns a handle that counts the number of activation from
    the output shape for the specified operation.

    Args:
        op_name (str): The name of the operation. If given, the handle will
            return a counter using this name.

    Returns:
        Callable: An activation handle for the given operation.
    """

    def _generic_activation_jit(
            i: Any, outputs: List[Any]) -> Union[typing.Counter[str], int]:
        """This is a generic jit handle that counts the number of activations
        for any operation given the output shape."""
        out_shape = get_shape(outputs[0])
        ac_count = prod(out_shape)  # type: ignore
        if op_name is None:
            return ac_count  # type: ignore
        else:
            return Counter({op_name: ac_count})

    return _generic_activation_jit


def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Union[int, Any]:
    """Count flops for fully connected layers."""
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]  # type: ignore
    assert len(input_shapes[1]) == 2, input_shapes[1]  # type: ignore
    batch_size, input_dim = input_shapes[0]  # type: ignore
    output_dim = input_shapes[1][1]  # type: ignore
    flops = batch_size * input_dim * output_dim
    return flops


def linear_flop_jit(inputs: List[Any], outputs: List[Any]) -> Union[int, Any]:
    """Count flops for the aten::linear operator."""
    # Inputs is a list of length 3; unlike aten::addmm, it is the first
    # two elements that are relevant.
    input_shapes = [get_shape(v) for v in inputs[0:2]]
    # input_shapes[0]: [dim0, dim1, ..., input_feature_dim]
    # input_shapes[1]: [output_feature_dim, input_feature_dim]
    assert input_shapes[0][-1] == input_shapes[1][-1]  # type: ignore
    flops = prod(input_shapes[0]) * input_shapes[1][0]  # type: ignore
    return flops


def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Union[int, Any]:
    """Count flops for the bmm operation."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]  # type: ignore
    d = input_shapes[-1][-1]  # type: ignore
    flop = n * c * t * d
    return flop


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Union[int, Any]:
    """Count flops for convolution. Note only multiplication is counted.
    Computation for addition and bias is ignored. Flops for a transposed
    convolution are calculated as.

    flops = (x_shape[2:] * prod(w_shape) * batch_size).

    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed

    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop


def conv_flop_jit(inputs: List[Any],
                  outputs: List[Any]) -> typing.Counter[str]:
    """Count flops for convolution."""
    # Inputs of Convolution should be a list of length 12 or 13.
    # They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    # starting with #40737 it will be 12) user_enabled_tf32
    assert len(inputs) == 12 or len(inputs) == 13, len(inputs)
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w),
                                   get_shape(outputs[0]))
    transposed = inputs[6].toIValue()

    # use a custom name instead of "_convolution"
    return Counter({
        'conv':
        conv_flop_count(
            x_shape,  # type: ignore
            w_shape,  # type: ignore
            out_shape,  # type: ignore
            transposed=transposed)  # type: ignore
    })


def einsum_flop_jit(inputs: List[Any], outputs: List[Any]) -> Union[int, Any]:
    """Count flops for the einsum operation."""
    # Inputs of einsum should be a list of length 2+.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    assert len(inputs) >= 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(' ', '')
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    if equation == 'abc,abd->acd':
        n, c, t = input_shapes[0]  # type: ignore
        p = input_shapes[-1][-1]  # type: ignore
        flop = n * c * t * p
        return flop

    elif equation == 'abc,adc->adb':
        n, t, g = input_shapes[0]  # type: ignore
        c = input_shapes[-1][1]  # type: ignore
        flop = n * t * g * c
        return flop
    else:
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize='optimal')[1]
        for line in optim.split('\n'):
            if 'optimized flop' in line.lower():
                # divided by 2 because we count MAC
                # (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(':')[-1]) / 2))
                return flop
        raise NotImplementedError('Unsupported einsum operation.')


def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Union[int, Any]:
    """Count flops for matmul."""
    # input_shapes is a list of length 2.
    input_shapes: list = [get_shape(v) for v in inputs]
    input1, input2 = input_shapes
    if len(input1) == 1:
        input1 = [1, input1[0]]
    if len(input2) == 1:
        input2 = [input2[0], 1]

    assert input1[-1] == input2[-2], input_shapes
    flop = prod(input1) * input2[-1]
    return flop


def norm_flop_counter(affine_arg_index: int) -> Handle:
    """
    Args:
        affine_arg_index: index of the affine argument in inputs
    """

    def norm_flop_jit(inputs: List[Any],
                      outputs: List[Any]) -> Union[int, Any]:
        """Count flops for norm layers."""
        # Inputs[0] contains the shape of the input.
        input_shape = get_shape(inputs[0])
        has_affine = get_shape(inputs[affine_arg_index]) is not None
        assert 2 <= len(input_shape) <= 5, input_shape  # type: ignore
        # 5 is just a rough estimate
        flop = prod(input_shape) * (5 if has_affine else 4)  # type: ignore
        return flop

    return norm_flop_jit


def batchnorm_flop_jit(inputs: List[Any],
                       outputs: List[Any]) -> Union[int, Any]:
    training = inputs[5].toIValue()
    assert isinstance(training,
                      bool), 'Signature of aten::batch_norm has changed!'
    if training:
        return norm_flop_counter(1)(inputs, outputs)  # pyre-ignore
    has_affine = get_shape(inputs[1]) is not None
    input_shape = prod(get_shape(inputs[0]))  # type: ignore
    return input_shape * (2 if has_affine else 1)


def elementwise_flop_counter(input_scale: float = 1,
                             output_scale: float = 0) -> Handle:
    """Count flops by.

        input_tensor.numel() * input_scale +
        output_tensor.numel() * output_scale

    Args:
        input_scale: scale of the input tensor (first argument)
        output_scale: scale of the output tensor (first element in outputs)
    """

    def elementwise_flop(inputs: List[Any],
                         outputs: List[Any]) -> Union[int, Any]:
        ret = 0
        if input_scale != 0:
            shape = get_shape(inputs[0])
            ret += input_scale * prod(shape)  # type: ignore
        if output_scale != 0:
            shape = get_shape(outputs[0])
            ret += output_scale * prod(shape)  # type: ignore
        return ret

    return elementwise_flop
