# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class DistSamplerSeedHook(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    priority = 'NORMAL'

    def before_train_epoch(self, runner) -> None:
        """Set the seed for sampler and batch_sampler.

        Args:
            runner (Runner): The runner of the training process.
        """
        if hasattr(runner.train_loop.dataloader, 'sampler') and hasattr(
                runner.train_loop.dataloader.sampler, 'set_epoch'):
            # In case the` _SingleProcessDataLoaderIter` has no sampler,
            # or data loader uses `SequentialSampler` in Pytorch.
            runner.train_loop.dataloader.sampler.set_epoch(runner.epoch)

        elif hasattr(runner.train_loop.dataloader,
                     'batch_sampler') and hasattr(
                         runner.train_loop.dataloader.batch_sampler.sampler,
                         'set_epoch'):
            # In case the` _SingleProcessDataLoaderIter` has no batch sampler.
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.train_loop.dataloader.batch_sampler.sampler.set_epoch(
                runner.epoch)
