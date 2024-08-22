# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.distributed.rpc import is_available

from mmengine.dist import is_main_process
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

try:
    from torch.distributed.optim import \
        ZeroRedundancyOptimizer as _ZeroRedundancyOptimizer
except ImportError:
    _ZeroRedundancyOptimizer = object

from .builder import OPTIMIZERS


@OPTIMIZERS.register_module()
class ZeroRedundancyOptimizer(_ZeroRedundancyOptimizer):
    """A wrapper class of :class:`ZeroRedundancyOptimizer` that gets a
    optimizer type as string.

    This class wraps an arbitrary :class:`torch.optim.Optimizer` and shards its
    states across ranks in the group as described by ZeRO_. The local optimizer
    instance in each rank is only responsible for updating approximately
    ``1 / world_size`` parameters and hence only needs to keep
    ``1 / world_size`` optimizer states. After parameters are updated locally,
    each rank will broadcast its parameters to all other peers to keep all
    model replicas in the same state. ``ZeroRedundancyOptimizer`` can be used
    in conjunction with :class:`torch.nn.parallel.DistributedDataParallel` to
    reduce per-rank peak memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Warnings:
        ``ZeroRedundancyOptimizer`` requires PyTorch >= 1.8.

    Warnings:
        ``ZeroRedundancyOptimizer`` requires PyTorch >= 1.12 to enable param
        groups.

    Args:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_type (str): the string of the local optimizer class.

    .. _ZeRO: https://arxiv.org/abs/1910.02054
    """

    def __init__(self, params, optimizer_type: str, **kwargs):
        assert digit_version(TORCH_VERSION) >= digit_version('1.8.0'), (
            '`torch.distributed.optim.ZeroReundancyOptimizer` is only '
            'available when pytorch version >= 1.8.')
        assert is_available(), 'torch.distributed.rpc is not available.'
        # Avoid the generator becoming empty after the following check
        params = list(params)
        assert (
            all(isinstance(p, torch.Tensor) for p in params)
            or digit_version(TORCH_VERSION) >= digit_version('1.12.0')), (
                'PyTorch ZeroRedundancyOptimizer started to support param '
                'groups since 1.12.0. Please update your pytorch version to '
                'enable this feature, or disable param groups by deleting '
                '`paramwise_cfg` filed in config file.')
        optimizer_class = getattr(torch.optim, optimizer_type)
        # TODO: Register a DDP communication hook for `overlap_with_ddp=True`.
        # Currently only `overlap_with_ddp=False` is supported. For more
        # details, please refer to the pytorch's official documentation.
        super().__init__(params, optimizer_class, **kwargs)

    def state_dict(self):
        """Consolidate `state_dict`s from ranks to save the `state_dict`."""
        self.consolidate_state_dict()
        state_dict = super().state_dict() if is_main_process() else dict()
        return state_dict
