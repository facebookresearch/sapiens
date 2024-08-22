# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from math import inf, isfinite
from typing import Optional, Tuple, Union

from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stop the training when the monitored metric reached a plateau.

    Args:
        monitor (str): The monitored metric key to decide early stopping.
        rule (str, optional): Comparison rule. Options are 'greater',
            'less'. Defaults to None.
        min_delta (float, optional): Minimum difference to continue the
            training. Defaults to 0.01.
        strict (bool, optional): Whether to crash the training when `monitor`
            is not found in the `metrics`. Defaults to False.
        check_finite: Whether to stop training when the monitor becomes NaN or
            infinite. Defaults to True.
        patience (int, optional): The times of validation with no improvement
            after which training will be stopped. Defaults to 5.
        stopping_threshold (float, optional): Stop training immediately once
            the monitored quantity reaches this threshold. Defaults to None.

    Note:
        `New in version 0.7.0.`
    """
    priority = 'LOWEST'

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(
        self,
        monitor: str,
        rule: Optional[str] = None,
        min_delta: float = 0.1,
        strict: bool = False,
        check_finite: bool = True,
        patience: int = 5,
        stopping_threshold: Optional[float] = None,
    ):

        self.monitor = monitor
        if rule is not None:
            if rule not in ['greater', 'less']:
                raise ValueError(
                    '`rule` should be either "greater" or "less", '
                    f'but got {rule}')
        else:
            rule = self._init_rule(monitor)
        self.rule = rule
        self.min_delta = min_delta if rule == 'greater' else -1 * min_delta
        self.strict = strict
        self.check_finite = check_finite
        self.patience = patience
        self.stopping_threshold = stopping_threshold

        self.wait_count = 0
        self.best_score = -inf if rule == 'greater' else inf

    def _init_rule(self, monitor: str) -> str:
        greater_keys = {key.lower() for key in self._default_greater_keys}
        less_keys = {key.lower() for key in self._default_less_keys}
        monitor_lc = monitor.lower()
        if monitor_lc in greater_keys:
            rule = 'greater'
        elif monitor_lc in less_keys:
            rule = 'less'
        elif any(key in monitor_lc for key in greater_keys):
            rule = 'greater'
        elif any(key in monitor_lc for key in less_keys):
            rule = 'less'
        else:
            raise ValueError(f'Cannot infer the rule for {monitor}, thus rule '
                             'must be specified.')
        return rule

    def _check_stop_condition(self, current_score: float) -> Tuple[bool, str]:
        compare = self.rule_map[self.rule]
        stop_training = False
        reason_message = ''

        if self.check_finite and not isfinite(current_score):
            stop_training = True
            reason_message = (f'Monitored metric {self.monitor} = '
                              f'{current_score} is infinite. '
                              f'Previous best value was '
                              f'{self.best_score:.3f}.')

        elif self.stopping_threshold is not None and compare(
                current_score, self.stopping_threshold):
            stop_training = True
            self.best_score = current_score
            reason_message = (f'Stopping threshold reached: '
                              f'`{self.monitor}` = {current_score} is '
                              f'{self.rule} than {self.stopping_threshold}.')
        elif compare(self.best_score + self.min_delta, current_score):

            self.wait_count += 1

            if self.wait_count >= self.patience:
                reason_message = (f'the monitored metric did not improve '
                                  f'in the last {self.wait_count} records. '
                                  f'best score: {self.best_score:.3f}. ')
                stop_training = True
        else:
            self.best_score = current_score
            self.wait_count = 0

        return stop_training, reason_message

    def before_run(self, runner) -> None:
        """Check `stop_training` variable in `runner.train_loop`.

        Args:
            runner (Runner): The runner of the training process.
        """

        assert hasattr(runner.train_loop, 'stop_training'), \
            '`train_loop` should contain `stop_training` variable.'

    def after_val_epoch(self, runner, metrics):
        """Decide whether to stop the training process.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """

        if self.monitor not in metrics:
            if self.strict:
                raise RuntimeError(
                    'Early stopping conditioned on metric '
                    f'`{self.monitor} is not available. Please check available'
                    f' metrics {metrics}, or set `strict=False` in '
                    '`EarlyStoppingHook`.')
            warnings.warn(
                'Skip early stopping process since the evaluation '
                f'results ({metrics.keys()}) do not include `monitor` '
                f'({self.monitor}).')
            return

        current_score = metrics[self.monitor]

        stop_training, message = self._check_stop_condition(current_score)
        if stop_training:
            runner.train_loop.stop_training = True
            runner.logger.info(message)
