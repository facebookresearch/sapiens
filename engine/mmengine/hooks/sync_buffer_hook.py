# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.dist import all_reduce_params, is_distributed
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class SyncBuffersHook(Hook):
    """Synchronize model buffers such as running_mean and running_var in BN at
    the end of each epoch."""

    priority = 'NORMAL'

    def __init__(self) -> None:
        self.distributed = is_distributed()
        # A flag to mark whether synchronization has been done in
        # after_train_epoch
        self.called_in_train = False

    def before_val_epoch(self, runner) -> None:
        """All-reduce model buffers before each validation epoch.

        Synchronize the buffers before each validation if they have not been
        synchronized at the end of the previous training epoch. This method
        will be called when using IterBasedTrainLoop.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.distributed:
            if not self.called_in_train:
                all_reduce_params(runner.model.buffers(), op='mean')
            self.called_in_train = False

    def after_train_epoch(self, runner) -> None:
        """All-reduce model buffers at the end of each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.distributed:
            all_reduce_params(runner.model.buffers(), op='mean')
            self.called_in_train = True
