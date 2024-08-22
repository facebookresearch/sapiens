# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from mmengine.optim.scheduler import CosineAnnealingParamScheduler

from mmpretrain.registry import PARAM_SCHEDULERS


class WeightDecaySchedulerMixin:
    """A mixin class for learning rate schedulers."""

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer, 'weight_decay', *args, **kwargs)


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingWeightDecay(WeightDecaySchedulerMixin,
                                 CosineAnnealingParamScheduler):
    """Set the weight decay value of each parameter group using a cosine
    annealing schedule.

    If the weight decay was set to be 0 initially, the weight decay value will
    be 0 constantly during the training.
    """

    def _get_value(self) -> list:
        """Compute value using chainable form of the scheduler."""

        def _get_eta_min(base_value):
            if self.eta_min_ratio is None:
                return self.eta_min
            return base_value * self.eta_min_ratio

        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        elif (self.last_step - 1 - self.T_max) % (2 * self.T_max) == 0:
            weight_decay_value_list = []
            for base_value, group in zip(self.base_values,
                                         self.optimizer.param_groups):
                if base_value == 0:
                    group_value = 0
                else:
                    group_value = group[self.param_name] + (
                        base_value - _get_eta_min(base_value)) * (
                            1 - math.cos(math.pi / self.T_max)) / 2
                weight_decay_value_list.append(group_value)
            return weight_decay_value_list

        weight_decay_value_list = []
        for base_value, group in zip(self.base_values,
                                     self.optimizer.param_groups):
            if base_value == 0:
                group_value = 0
            else:
                group_value = (
                    1 + math.cos(math.pi * self.last_step / self.T_max)) / (
                        1 + math.cos(math.pi *
                                     (self.last_step - 1) / self.T_max)
                    ) * (group[self.param_name] -
                         _get_eta_min(base_value)) + _get_eta_min(base_value)
            weight_decay_value_list.append(group_value)
        return weight_decay_value_list
