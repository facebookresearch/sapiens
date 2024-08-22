# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import operator as op
from typing import Any, Optional, Union

from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS
from mmpretrain.utils import get_ori_model


@HOOKS.register_module()
class WarmupParamHook(Hook):
    """This is a hook used for changing the parameters other than optimizations
    that need to warmup inside the module.

    This hook can extend with more detailed warmup rule if necessary.

    Args:
        param_name (str): The parameter name that needs to be altered.
        module_name (str): Module name that belongs to the model. Such as
            `head`, `head.loss`, etc.
        warmup_epochs (int): The warmup epochs for this parameter.
    """

    def __init__(
        self,
        param_name: str,
        module_name: str,
        warmup_epochs: int,
    ) -> None:
        self.param_name = param_name
        self.warmup_epochs = warmup_epochs
        # getter for module which saves the changed parameter
        self.module_getter = op.attrgetter(module_name)

    def get_param(self, runner) -> Any:
        """Get the parameter."""
        try:
            module = self.module_getter(get_ori_model(runner.model))
            return getattr(module, self.param_name)
        except AttributeError as e:
            raise AttributeError(f'{e}. Please check hook settings.')

    def set_param(self, runner, value) -> None:
        """Set the parameter."""
        try:
            module = self.module_getter(get_ori_model(runner.model))
            setattr(module, self.param_name, value)
        except AttributeError as e:
            raise AttributeError(f'{e}. Please check hook settings.')

    def before_train(self, runner) -> None:
        """Get the original value before train."""
        self.ori_val = self.get_param(runner)

    def before_train_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: Optional[Union[dict, tuple, list]] = None) -> None:
        """Set the warmup value before each train iter."""
        cur_iter = runner.iter
        iters_per_epoch = runner.max_iters / runner.max_epochs
        new_val = self.ori_val * min(
            1, cur_iter / (self.warmup_epochs * iters_per_epoch))
        self.set_param(runner, new_val)
