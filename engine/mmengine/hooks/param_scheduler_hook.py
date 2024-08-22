# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Union

from mmengine.optim import _ParamScheduler
from mmengine.registry import HOOKS
from mmengine.utils import is_list_of
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class ParamSchedulerHook(Hook):
    """A hook to update some hyper-parameters in optimizer, e.g., learning rate
    and momentum."""

    priority = 'LOW'

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Call step function for each scheduler after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                In order to keep this interface consistent with other hooks,
                we keep ``data_batch`` here.
            outputs (dict, optional): Outputs from model.
                In order to keep this interface consistent with other hooks, we
                keep ``data_batch`` here.
        """

        if runner.param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if not scheduler.by_epoch:
                    scheduler.step()

        if isinstance(runner.param_schedulers, list):
            step(runner.param_schedulers)
        elif isinstance(runner.param_schedulers, dict):
            for param_schedulers in runner.param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {runner.param_schedulers}')

    def after_train_epoch(self, runner) -> None:
        """Call step function for each scheduler after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """

        if runner.param_schedulers is None:
            return

        def step(param_schedulers):
            assert isinstance(param_schedulers, list)
            for scheduler in param_schedulers:
                if scheduler.by_epoch:
                    scheduler.step()

        if isinstance(runner.param_schedulers, list):
            step(runner.param_schedulers)
        elif isinstance(runner.param_schedulers, dict):
            for param_schedulers in runner.param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {runner.param_schedulers}')

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Call step function for each scheduler which has attribute
        ``need_val_args`` after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.

        Note:
            if ``runner.param_schedulers`` is not built before,
            the hook ``after_val_epoch`` will be skipped.
        """

        if runner.param_schedulers is None:
            return

        # avoid counting scheduler._global_step
        # it has counted in after_train_* hook
        if metrics is None:
            return

        def step(param_schedulers):
            # check param_schedulers is list and built
            if not is_list_of(param_schedulers, _ParamScheduler):
                return

            for scheduler in param_schedulers:
                if (scheduler.by_epoch
                        and getattr(scheduler, 'need_val_args', False)):
                    scheduler.step(metrics)

        if isinstance(runner.param_schedulers, list):
            step(runner.param_schedulers)
        elif isinstance(runner.param_schedulers, dict):
            for param_schedulers in runner.param_schedulers.values():
                step(param_schedulers)
        else:
            raise TypeError(
                'runner.param_schedulers should be list of ParamScheduler or '
                'a dict containing list of ParamScheduler, '
                f'but got {runner.param_schedulers}')
