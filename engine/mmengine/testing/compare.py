# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Iterable
from runpy import run_path
from shlex import split
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import patch

from torch.nn import GroupNorm, LayerNorm
from torch.testing import assert_allclose as _assert_allclose

from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm


def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = True,
    msg: Optional[Union[str, Callable]] = '',
) -> None:
    """Asserts that ``actual`` and ``expected`` are close. A wrapper function
    of ``torch.testing.assert_allclose``.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must
            also be specified. If omitted, default values based on the
            :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol`
            must also be specified. If omitted, default values based on the
            :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (bool): If ``True``, two ``NaN`` values will be considered
            equal.
        msg (Optional[Union[str, Callable]]): Optional error message to use if
            the values of corresponding tensors mismatch. Unused when PyTorch
            < 1.6.
    """
    if 'parrots' not in TORCH_VERSION and \
            digit_version(TORCH_VERSION) >= digit_version('1.6'):
        _assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=msg)
    else:
        # torch.testing.assert_allclose has no ``msg`` argument
        # when PyTorch < 1.6
        _assert_allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)


def check_python_script(cmd):
    """Run the python cmd script with `__main__`. The difference between
    `os.system` is that, this function exectues code in the current process, so
    that it can be tracked by coverage tools. Currently it supports two forms:

    - ./tests/data/scripts/hello.py zz
    - python tests/data/scripts/hello.py zz
    """
    args = split(cmd)
    if args[0] == 'python':
        args = args[1:]
    with patch.object(sys, 'argv', args):
        run_path(args[0], run_name='__main__')


def _any(judge_result):
    """Since built-in ``any`` works only when the element of iterable is not
    iterable, implement the function."""
    if not isinstance(judge_result, Iterable):
        return judge_result

    try:
        for element in judge_result:
            if _any(element):
                return True
    except TypeError:
        # Maybe encounter the case: torch.tensor(True) | torch.tensor(False)
        if judge_result:
            return True
    return False


def assert_dict_contains_subset(dict_obj: Dict[Any, Any],
                                expected_subset: Dict[Any, Any]) -> bool:
    """Check if the dict_obj contains the expected_subset.

    Args:
        dict_obj (Dict[Any, Any]): Dict object to be checked.
        expected_subset (Dict[Any, Any]): Subset expected to be contained in
            dict_obj.

    Returns:
        bool: Whether the dict_obj contains the expected_subset.
    """

    for key, value in expected_subset.items():
        if key not in dict_obj.keys() or _any(dict_obj[key] != value):
            return False
    return True


def assert_attrs_equal(obj: Any, expected_attrs: Dict[str, Any]) -> bool:
    """Check if attribute of class object is correct.

    Args:
        obj (object): Class object to be checked.
        expected_attrs (Dict[str, Any]): Dict of the expected attrs.

    Returns:
        bool: Whether the attribute of class object is correct.
    """
    for attr, value in expected_attrs.items():
        if not hasattr(obj, attr) or _any(getattr(obj, attr) != value):
            return False
    return True


def assert_dict_has_keys(obj: Dict[str, Any],
                         expected_keys: List[str]) -> bool:
    """Check if the obj has all the expected_keys.

    Args:
        obj (Dict[str, Any]): Object to be checked.
        expected_keys (List[str]): Keys expected to contained in the keys of
            the obj.

    Returns:
        bool: Whether the obj has the expected keys.
    """
    return set(expected_keys).issubset(set(obj.keys()))


def assert_keys_equal(result_keys: List[str], target_keys: List[str]) -> bool:
    """Check if target_keys is equal to result_keys.

    Args:
        result_keys (List[str]): Result keys to be checked.
        target_keys (List[str]): Target keys to be checked.

    Returns:
        bool: Whether target_keys is equal to result_keys.
    """
    return set(result_keys) == set(target_keys)


def assert_is_norm_layer(module) -> bool:
    """Check if the module is a norm layer.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: Whether the module is a norm layer.
    """

    norm_layer_candidates = (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm)
    return isinstance(module, norm_layer_candidates)


def assert_params_all_zeros(module) -> bool:
    """Check if the parameters of the module is all zeros.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: Whether the parameters of the module is all zeros.
    """
    weight_data = module.weight.data
    is_weight_zero = weight_data.allclose(
        weight_data.new_zeros(weight_data.size()))

    if hasattr(module, 'bias') and module.bias is not None:
        bias_data = module.bias.data
        is_bias_zero = bias_data.allclose(
            bias_data.new_zeros(bias_data.size()))
    else:
        is_bias_zero = True

    return is_weight_zero and is_bias_zero
