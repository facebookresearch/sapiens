# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.logging import print_log
from mmengine.utils.dl_utils import mmcv_full_available


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def detect_anomalous_params(loss: torch.Tensor, model) -> None:
    parameters_in_graph = set()
    visited = set()

    def traverse(grad_fn):
        if grad_fn is None:
            return
        if grad_fn not in visited:
            visited.add(grad_fn)
            if hasattr(grad_fn, 'variable'):
                parameters_in_graph.add(grad_fn.variable)
            parents = grad_fn.next_functions
            if parents is not None:
                for parent in parents:
                    grad_fn = parent[0]
                    traverse(grad_fn)

    traverse(loss.grad_fn)
    for n, p in model.named_parameters():
        if p not in parameters_in_graph and p.requires_grad:
            print_log(
                f'{n} with shape {p.size()} is not '
                f'in the computational graph \n',
                logger='current',
                level=logging.ERROR)


def merge_dict(*args):
    """Merge all dictionaries into one dictionary.

    If pytorch version >= 1.8, ``merge_dict`` will be wrapped
    by ``torch.fx.wrap``,  which will make ``torch.fx.symbolic_trace`` skip
    trace ``merge_dict``.

    Note:
        If a function needs to be traced by ``torch.fx.symbolic_trace``,
        but inevitably needs to use ``update`` method of ``dict``(``update``
        is not traceable). It should use ``merge_dict`` to replace
        ``xxx.update``.

    Args:
        *args: dictionary needs to be merged.

    Returns:
        dict: Merged dict from args
    """
    output = dict()
    for item in args:
        assert isinstance(
            item,
            dict), (f'all arguments of merge_dict should be a dict, but got '
                    f'{type(item)}')
        output.update(item)
    return output


# torch.fx is only available when pytorch version >= 1.8.
# If the subclass of `BaseModel` has multiple submodules, and each module
# will return a loss dict during training process, i.e., `TwoStageDetector`
# in mmdet. It should use `merge_dict` to get the total loss, rather than
# `loss.update` to keep model traceable.
try:
    import torch.fx

    # make torch.fx skip trace `merge_dict`.
    merge_dict = torch.fx.wrap(merge_dict)

except ImportError:
    warnings.warn('Cannot import torch.fx, `merge_dict` is a simple function '
                  'to merge multiple dicts')


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input: torch.Tensor):
        return


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if mmcv_full_available():
        from mmcv.ops import SyncBatchNorm
        module_checklist.append(SyncBatchNorm)

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps,
                                     module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        # Some custom modules or 3rd party implemented modules may raise an
        # error when calling `add_module`. Therefore, try to catch the error
        # and do not raise it. See https://github.com/open-mmlab/mmengine/issues/638 # noqa: E501
        # for more details.
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print_log(
                F'Failed to convert {child} from SyncBN to BN!',
                logger='current',
                level=logging.WARNING)
    del module
    return module_output


def convert_sync_batchnorm(module: nn.Module,
                           implementation='torch') -> nn.Module:
    """Helper function to convert all `BatchNorm` layers in the model to
    `SyncBatchNorm` (SyncBN) or `mmcv.ops.sync_bn.SyncBatchNorm` (MMSyncBN)
    layers. Adapted from `PyTorch convert sync batchnorm`_.

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.
        implementation (str): The type of `SyncBatchNorm` to convert to.

            - 'torch': convert to `torch.nn.modules.batchnorm.SyncBatchNorm`.
            - 'mmcv': convert to `mmcv.ops.sync_bn.SyncBatchNorm`.

    Returns:
        nn.Module: The converted module with `SyncBatchNorm` layers.

    .. _PyTorch convert sync batchnorm:
       https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm
    """  # noqa: E501
    module_output = module

    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if implementation == 'torch':
            SyncBatchNorm = torch.nn.modules.batchnorm.SyncBatchNorm
        elif implementation == 'mmcv':
            from mmcv.ops import SyncBatchNorm  # type: ignore
        else:
            raise ValueError('sync_bn should be "torch" or "mmcv", but got '
                             f'{implementation}')

        module_output = SyncBatchNorm(module.num_features, module.eps,
                                      module.momentum, module.affine,
                                      module.track_running_stats)

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name,
                                 convert_sync_batchnorm(child, implementation))
    del module
    return module_output
