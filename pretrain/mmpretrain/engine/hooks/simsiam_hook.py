# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class SimSiamHook(Hook):
    """Hook for SimSiam.

    This hook is for SimSiam to fix learning rate of predictor.

    Args:
        fix_pred_lr (bool): whether to fix the lr of predictor or not.
        lr (float): the value of fixed lr.
        adjust_by_epoch (bool, optional): whether to set lr by epoch or iter.
            Defaults to True.
    """

    def __init__(self,
                 fix_pred_lr: bool,
                 lr: float,
                 adjust_by_epoch: Optional[bool] = True) -> None:
        self.fix_pred_lr = fix_pred_lr
        self.lr = lr
        self.adjust_by_epoch = adjust_by_epoch

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        """fix lr of predictor by iter."""
        if self.adjust_by_epoch:
            return
        else:
            if self.fix_pred_lr:
                for param_group in runner.optim_wrapper.optimizer.param_groups:
                    if 'fix_lr' in param_group and param_group['fix_lr']:
                        param_group['lr'] = self.lr

    def before_train_epoch(self, runner) -> None:
        """fix lr of predictor by epoch."""
        if self.fix_pred_lr:
            for param_group in runner.optim_wrapper.optimizer.param_groups:
                if 'fix_lr' in param_group and param_group['fix_lr']:
                    param_group['lr'] = self.lr
