# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.registry import PARAM_SCHEDULERS
# yapf: disable
from .param_scheduler import (ConstantParamScheduler,
                              CosineAnnealingParamScheduler,
                              CosineRestartParamScheduler,
                              ExponentialParamScheduler, LinearParamScheduler,
                              MultiStepParamScheduler, PolyParamScheduler,
                              ReduceOnPlateauParamScheduler,
                              StepParamScheduler)

# yapf: enable


class MomentumSchedulerMixin:
    """A mixin class for momentum schedulers.

    It can schedule the momentum in SGD and the beta_0 in Adam series.
    """

    def __init__(self, optimizer, *args, **kwargs):
        self.use_betas = False
        if 'momentum' in optimizer.defaults:
            param_name = 'momentum'
        elif 'betas' in optimizer.defaults:
            # for Adam series optimizer, the momentum is beta_0
            self.use_betas = True
            param_name = 'momentum'
            for group in optimizer.param_groups:
                # set a reference momentum in the param groups for scheduling
                group[param_name] = group['betas'][0]
        else:
            raise ValueError(
                'optimizer must support momentum when using momentum scheduler'
            )
        super().__init__(optimizer, param_name, *args, **kwargs)

    def step(self):
        """Adjusts the momentum of each parameter group based on the specified
        schedule."""
        super().step()
        if self.use_betas:
            for group in self.optimizer.param_groups:
                _, beta_1 = group['betas']
                # update the betas with the calculated value
                group['betas'] = (group['momentum'], beta_1)


@PARAM_SCHEDULERS.register_module()
class ConstantMomentum(MomentumSchedulerMixin, ConstantParamScheduler):
    """Decays the momentum value of each parameter group by a small constant
    factor until the number of epoch reaches a pre-defined milestone: ``end``.
    Notice that such decay can happen simultaneously with other changes to the
    momentum value from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        factor (float): The number we multiply momentum until the milestone.
            Defaults to 1./3.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without state
            dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by epochs.
            Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingMomentum(MomentumSchedulerMixin,
                              CosineAnnealingParamScheduler):
    r"""Set the momentum of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial value and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    Notice that because the schedule
    is defined recursively, the momentum can be simultaneously modified
    outside this scheduler by other operators. If the momentum is set
    solely by this scheduler, the momentum at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum momentum value. Defaults to None.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
        eta_min_ratio (float, optional): The ratio of the minimum parameter
            value to the base parameter value. Either `eta_min` or
            `eta_min_ratio` should be specified. Defaults to None.
            New in version 0.3.2.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """


@PARAM_SCHEDULERS.register_module()
class ExponentialMomentum(MomentumSchedulerMixin, ExponentialParamScheduler):
    """Decays the momentum of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        gamma (float): Multiplicative factor of momentum value decay.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class LinearMomentum(MomentumSchedulerMixin, LinearParamScheduler):
    """Decays the momentum of each parameter group by linearly changing
    small multiplicative factor until the number of epoch reaches a pre-defined
    milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    momentum from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        start_factor (float): The number we multiply momentum in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 1./3.
        end_factor (float): The number we multiply momentum at the end
            of linear changing process. Defaults to 1.0.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class MultiStepMomentum(MomentumSchedulerMixin, MultiStepParamScheduler):
    """Decays the specified momentum in each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the momentum from outside this
    scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of momentum value decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class StepMomentum(MomentumSchedulerMixin, StepParamScheduler):
    """Decays the momentum of each parameter group by gamma every step_size
    epochs. Notice that such decay can happen simultaneously with other changes
    to the momentum from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        step_size (int): Period of momentum value decay.
        gamma (float): Multiplicative factor of momentum value decay.
            Defaults to 0.1.
        begin (int): Step at which to start updating the momentum.
            Defaults to 0.
        end (int): Step at which to stop updating the momentum.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled momentum is updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the momentum for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class PolyMomentum(MomentumSchedulerMixin, PolyParamScheduler):
    """Decays the momentum of each parameter group in a polynomial decay
    scheme.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        eta_min (float): Minimum momentum at the end of scheduling.
            Defaults to 0.
        power (float): The power of the polynomial. Defaults to 1.0.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class CosineRestartMomentum(MomentumSchedulerMixin,
                            CosineRestartParamScheduler):
    """Sets the momentum of each parameter group according to the cosine
    annealing with restarts scheme. The cosine restart policy anneals the
    momentum from the initial value to `eta_min` with a cosine annealing
    schedule and then restarts another period from the maximum value multiplied
    with `restart_weight`.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float]): Restart weights at each
            restart iteration. Defaults to [1].
        eta_min (float): Minimum parameter value at the end of scheduling.
            Defaults to None.
        eta_min_ratio (float, optional): The ratio of minimum parameter value
            to the base parameter value. Either `min_lr` or `min_lr_ratio`
            should be specified. Defaults to None.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class ReduceOnPlateauMomentum(MomentumSchedulerMixin,
                              ReduceOnPlateauParamScheduler):
    """Reduce the momentum of each parameter group when a metric has stopped
    improving. Models often benefit from reducing the momentum by a factor of
    2-10 once learning stagnates. This scheduler reads a metrics quantity and
    if no improvement is seen for a ``patience`` number of epochs, the momentum
    is reduced.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        monitor (str): Key name of the value to monitor in metrics dict.
        rule (str): One of `less`, `greater`. In `less` rule, momentum will
            be reduced when the quantity monitored has stopped
            decreasing; in `greater` rule it will be reduced when the
            quantity monitored has stopped increasing. Defaults to 'less'.
            The ``rule`` is the renaming of ``mode`` in pytorch.
        factor (float): Factor by which the momentum will be
            reduced. new_param = param * factor. Defaults to 0.1.
        patience (int): Number of epochs with no improvement after
            which momentum will be reduced. For example, if
            ``patience = 2``, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the momentum after
            the 3rd epoch if the monitor value still hasn't improved then.
            Defaults to 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Defaults to 1e-4.
        threshold_rule (str): One of `rel`, `abs`. In `rel` rule,
            dynamic_threshold = best * ( 1 + threshold ) in 'greater'
            rule or best * ( 1 - threshold ) in `less` rule.
            In `abs` rule, dynamic_threshold = best + threshold in
            `greater` rule or best - threshold in `less` rule.
            Defaults to 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after momentum has been reduced. Defaults to 0.
        min_value (float or list[float]): A scalar or a sequence of scalars.
            A lower bound on the momentum of each parameter group
            respectively. Defaults to 0. .
        eps (float): Minimal decay applied to momentum. If the difference
            between new and old momentum is smaller than eps, the update is
            ignored. Defaults to 1e-8.
        begin (int): Step at which to start triggering the scheduler
            to monitor in val within the interval calculated
            according to epoch of training. Defaults to 0.
        end (int): Step at which to stop triggering the scheduler
            to monitor in val within the interval calculated
            according to epoch of training. Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """

    def step(self, metrics=None):
        """Adjusts the momentum of each parameter group based on the specified
        schedule.

        Args:
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
                Defaults to None.
        """
        super(MomentumSchedulerMixin, self).step(metrics)
        if self.use_betas:
            for group in self.optimizer.param_groups:
                _, beta_1 = group['betas']
                # update the betas with the calculated value
                group['betas'] = (group['momentum'], beta_1)
