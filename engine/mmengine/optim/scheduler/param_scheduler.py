# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ------------------------------------------------------------------------
# Modified from https://github.com/pytorch/pytorch
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math
import warnings
import weakref
from collections import Counter
from functools import wraps
from typing import Callable, List, Optional, Sequence, Union

from torch.optim import Optimizer

from mmengine.logging import print_log
from mmengine.optim import BaseOptimWrapper
from mmengine.registry import PARAM_SCHEDULERS

INF = int(1e9)

OptimizerType = Union[BaseOptimWrapper, Optimizer]


class _ParamScheduler:
    """Base class for parameter schedulers.

    It should be inherited by all schedulers that schedule parameters in the
    optimizer's ``param_groups``. All subclasses should overwrite the
    ``_get_value()`` according to their own schedule strategy.
    The implementation is motivated by
    https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.

    Args:
        optimizer (BaseOptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resuming without
            state dict. Default value ``-1`` means the ``step`` function is
            never be called before. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """  # noqa: E501

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, BaseOptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))
        self.base_values = [
            group[f'initial_{param_name}'] for group in optimizer.param_groups
        ]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method: Callable):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)  # type: ignore
            # Get the unbound method for the same purpose.
            func = method.__func__  # type: ignore
            cls = instance_ref().__class__  # type: ignore
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._global_step += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)  # type: ignore
        self.optimizer._global_step = -1  # type: ignore

        self._global_step = -1
        self.verbose = verbose

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not
        the optimizer.

        Returns:
            dict: scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler.

        Returns:
            list: A list of the last computed value of the optimizer's
            ``param_group``.
        """
        return self._last_value

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        raise NotImplementedError

    def print_value(self, is_verbose: bool, group: int, value: float):
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e}.',
                logger='current')

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after '
                    'parameter value scheduler initialization. Please, make '
                    'sure to call `optimizer.step()` before '
                    '`scheduler.step()`. See more details at '
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before '
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you '
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class StepParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the parameter value from outside this scheduler.

    Args:
        optimizer (BaseOptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        step_size (int): Period of parameter value decay.
        gamma (float): Multiplicative factor of parameter value decay.
            Defaults to 0.1.
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

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 step_size: int,
                 gamma: float = 0.1,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              step_size,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        step_size = step_size * epoch_length
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            step_size=step_size,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if (self.last_step == 0) or (self.last_step % self.step_size != 0):
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] * self.gamma
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class MultiStepParamScheduler(_ParamScheduler):
    """Decays the specified parameter in each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the parameter from outside this
    scheduler.

    Args:
        optimizer (BaseOptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of parameter value decay.
            Defaults to 0.1.
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

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 milestones: List[int],
                 gamma: float = 0.1,
                 last_step: int = -1,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              milestones,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        milestones = [i * epoch_length for i in milestones]
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            milestones=milestones,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step not in self.milestones:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] *
            self.gamma**self.milestones[self.last_step]
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class ConstantParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by a small constant
    factor until the number of epoch reaches a pre-defined milestone: ``end``.
    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        factor (float): The number we multiply parameter value until the
            milestone. Defaults to 1./3.
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

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 factor: float = 1.0 / 3,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        if factor > 1.0 or factor < 0:
            raise ValueError(
                'Constant multiplicative factor should between 0 and 1.')

        self.factor = factor
        self.total_iters = end - begin - 1
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name] * self.factor
                for group in self.optimizer.param_groups
            ]

        if (self.last_step > self.total_iters
                or (self.last_step != self.total_iters)):
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        if self.last_step == self.total_iters:
            return [
                group[self.param_name] * (1.0 / self.factor)
                for group in self.optimizer.param_groups
            ]


@PARAM_SCHEDULERS.register_module()
class ExponentialParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by gamma every epoch.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        gamma (float): Multiplicative factor of parameter value decay.
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

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 gamma: float,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.gamma = gamma
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        return [
            group[self.param_name] * self.gamma
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingParamScheduler(_ParamScheduler):
    r"""Set the parameter value of each parameter group using a cosine
    annealing schedule, where :math:`\eta_{max}` is set to the initial value
    and :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

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
    is defined recursively, the parameter value can be simultaneously modified
    outside this scheduler by other operators. If the parameter value is set
    solely by this scheduler, the parameter value at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this
    only implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        T_max (int, optional): Maximum number of iterations. If not specified,
            use ``end - begin``. Defaults to None.
        eta_min (float, optional): Minimum parameter value. Defaults to None.
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
        eta_min_ratio (float, optional): The ratio of the minimum parameter
            value to the base parameter value. Either `eta_min` or
            `eta_min_ratio` should be specified. Defaults to None.
            New in version 0.3.2.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 T_max: Optional[int] = None,
                 eta_min: Optional[float] = None,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False,
                 eta_min_ratio: Optional[float] = None):
        # To preserve backwards compatibility
        if eta_min is None and eta_min_ratio is None:
            eta_min = 0.
        assert (eta_min is None) ^ (eta_min_ratio is None), \
            'Either `eta_min` or `eta_min_ratio should be specified'
        self.T_max = T_max or (end - begin)
        self.eta_min = eta_min
        self.eta_min_ratio = eta_min_ratio
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              T_max=None,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        if T_max is not None:
            T_max = T_max * epoch_length
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            T_max=T_max,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

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
            return [
                group[self.param_name] +
                (base_value - _get_eta_min(base_value)) *
                (1 - math.cos(math.pi / self.T_max)) / 2
                for base_value, group in zip(self.base_values,
                                             self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * self.last_step / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_step - 1) / self.T_max)) *
                (group[self.param_name] - _get_eta_min(base_value)) +
                _get_eta_min(base_value) for base_value, group in zip(
                    self.base_values, self.optimizer.param_groups)]


@PARAM_SCHEDULERS.register_module()
class LinearParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by linearly changing
    small multiplicative factor until the number of epoch reaches a pre-defined
    milestone: ``end``.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        start_factor (float): The number we multiply parameter value in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 1./3.
        end_factor (float): The number we multiply parameter value at the end
            of linear changing process. Defaults to 1.0.
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

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 start_factor: float = 1.0 / 3,
                 end_factor: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                'Starting multiplicative factor should between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                'Ending multiplicative factor should between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = end - begin - 1
        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name] * self.start_factor
                for group in self.optimizer.param_groups
            ]

        return [
            group[self.param_name] *
            (1. + (self.end_factor - self.start_factor) /
             (self.total_iters * self.start_factor + (self.last_step - 1) *
              (self.end_factor - self.start_factor)))
            for group in self.optimizer.param_groups
        ]


@PARAM_SCHEDULERS.register_module()
class PolyParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group in a polynomial decay
    scheme.

    Notice that such decay can happen simultaneously with other changes to the
    parameter value from outside this scheduler.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        eta_min (float): Minimum parameter value at the end of scheduling.
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

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 eta_min: float = 0,
                 power: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        self.eta_min = eta_min
        self.power = power
        self.total_iters = end - begin - 1

        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]

        return [(group[self.param_name] - self.eta_min) *
                (1 - 1 / (self.total_iters - self.last_step + 1))**self.power +
                self.eta_min for group in self.optimizer.param_groups]


@PARAM_SCHEDULERS.register_module()
class OneCycleParamScheduler(_ParamScheduler):
    r"""Sets the parameters of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every
    batch. `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in
    one of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. If total_steps is not defined, begin and end of the ParamSchedul will
       works for it. In this case, the number of total steps is inferred by
       total_steps = end - begin

    The default behaviour of this scheduler follows the fastai implementation
    of 1cycle, which claims that "unpublished work has shown even better
    results by using only two phases". To mimic the behaviour of the original
    paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        eta_max (float or list): Upper parameter value boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it will be equal to
            ``end - begin``. Defaults to None
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Defaults to 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing,
            "linear" for linear annealing.
            Defaults to 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_param = eta_max/div_factor
            Defaults to 25
        final_div_factor (float): Determines the minimum learning rate via
            eta_min = initial_param/final_div_factor
            Defaults to 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to
            annihilate the learning rate according to 'final_div_factor'
            instead of modifying the second phase (the first two phases will be
            symmetrical about the step indicated by 'pct_start').
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """  # noqa E501

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 eta_max: float = 0,
                 total_steps: Optional[int] = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.,
                 final_div_factor: float = 1e4,
                 three_phase: bool = False,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        assert param_name == 'lr', ('OneCycle only works for learning rate '
                                    'updating, but got patam_name as '
                                    f'{param_name}')

        self.eta_max = eta_max
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Validate total_steps
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError('Expected positive integer total_steps, '
                                 f'but got {total_steps}')
            self.total_steps = total_steps
        else:
            self.total_steps = self.end - self.begin

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError('Expected float between 0 and 1 pct_start, '
                             f'but got {pct_start}')

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError(
                'anneal_strategy must by one of "cos" or "linear", '
                f'instead got {anneal_strategy}')
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        if three_phase:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * self.total_steps) - 1,
                    f'start_{param_name}': f'initial_{param_name}',
                    f'end_{param_name}': f'max_{param_name}'
                },
                {
                    'end_step': float(2 * pct_start * self.total_steps) - 2,
                    f'start_{param_name}': f'max_{param_name}',
                    f'end_{param_name}': f'initial_{param_name}'
                },
                {
                    'end_step': self.total_steps - 1,
                    f'start_{param_name}': f'initial_{param_name}',
                    f'end_{param_name}': f'min_{param_name}'
                },
            ]
        else:
            self._schedule_phases = [
                {
                    'end_step': float(pct_start * self.total_steps) - 1,
                    f'start_{param_name}': f'initial_{param_name}',
                    f'end_{param_name}': f'max_{param_name}'
                },
                {
                    'end_step': self.total_steps - 1,
                    f'start_{param_name}': f'max_{param_name}',
                    f'end_{param_name}': f'min_{param_name}'
                },
            ]

        # Initialize parameters
        max_values = self._format_param(f'max_{param_name}', optimizer,
                                        eta_max)
        if last_step == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group[f'initial_{param_name}'] = max_values[idx] / div_factor
                group[f'max_{param_name}'] = max_values[idx]
                group[f'min_{param_name}'] = \
                    group[f'initial_{param_name}'] / final_div_factor

        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    f'expected {len(optimizer.param_groups)} values '
                    f'for {name}, got {len(param)}')
            return param
        else:
            return [param] * len(optimizer.param_groups)

    @staticmethod
    def _annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""

        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to
        1.0."""
        return (end - start) * pct + start

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              total_steps=None,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        if total_steps is not None:
            total_steps = total_steps * epoch_length
        return cls(
            *args,
            begin=begin,
            end=end,
            total_steps=total_steps,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""

        params = []
        step_num = self.last_step

        if step_num > self.total_steps:
            raise ValueError(
                f'Tried to step {step_num + 1} times. '
                f'The specified number of total steps is {self.total_steps}')

        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase['end_step']
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_param = self.anneal_func(
                        group[phase['start_' + self.param_name]],
                        group[phase['end_' + self.param_name]], pct)
                    break
                start_step = phase['end_step']

            params.append(computed_param)

        return params


@PARAM_SCHEDULERS.register_module()
class CosineRestartParamScheduler(_ParamScheduler):
    """Sets the parameters of each parameter group according to the cosine
    annealing with restarts scheme. The cosine restart policy anneals the
    parameter from the initial value to `eta_min` with a cosine annealing
    schedule and then restarts another period from the maximum value multiplied
    with `restart_weight`.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float]): Restart weights at each
            restart iteration. Defaults to [1].
        eta_min (float, optional): Minimum parameter value at the end of
            scheduling. Defaults to None.
        eta_min_ratio (float, optional): The ratio of minimum parameter value
            to the base parameter value. Either `eta_min` or `eta_min_ratio`
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

    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 periods: List[int],
                 restart_weights: Sequence[float] = (1, ),
                 eta_min: Optional[float] = None,
                 eta_min_ratio: Optional[float] = None,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        assert (eta_min is None) ^ (eta_min_ratio is None)
        self.periods = periods
        self.eta_min = eta_min
        self.eta_min_ratio = eta_min_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        super().__init__(
            optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              periods,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        periods = [p * epoch_length for p in periods]
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            periods=periods,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        idx = self.get_position_from_periods(self.last_step,
                                             self.cumulative_periods)
        # if current step is not in the periods, return origin parameters
        if idx is None:
            return [
                group[self.param_name] for group in self.optimizer.param_groups
            ]
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]
        step = self.last_step - nearest_restart
        values = []
        for base_value, group in zip(self.base_values,
                                     self.optimizer.param_groups):
            eta_max = base_value * current_weight
            if self.eta_min_ratio is None:
                eta_min = self.eta_min
            else:
                eta_min = base_value * self.eta_min_ratio
            if step == 0:
                values.append(eta_max)
            else:
                values.append(
                    (1 + math.cos(math.pi * step / current_periods)) /
                    (1 + math.cos(math.pi * (step - 1) / current_periods)) *
                    (group[self.param_name] - eta_min) + eta_min)

        return values

    @staticmethod
    def get_position_from_periods(
            iteration: int, cumulative_periods: List[int]) -> Optional[int]:
        """Get the position from a period list.

        It will return the index of the right-closest number in the period
        list.
        For example, the cumulative_periods = [100, 200, 300, 400],
        if iteration == 50, return 0;
        if iteration == 210, return 2;
        if iteration == 300, return 3.

        Args:
            iteration (int): Current iteration.
            cumulative_periods (list[int]): Cumulative period list.

        Returns:
            Optional[int]: The position of the right-closest number in the
            period list. If not in the period, return None.
        """
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        return None


@PARAM_SCHEDULERS.register_module()
class ReduceOnPlateauParamScheduler(_ParamScheduler):
    """Reduce the parameters of each parameter group when a metric has stopped
    improving. Models often benefit from reducing the parameters by a factor of
    2-10 once learning stagnates. This scheduler reads a metrics quantity and
    if no improvement is seen for a ``patience`` number of epochs, the
    parameters are reduced.

    The implementation is motivated by `PyTorch ReduceLROnPlateau`_.

    Args:
        optimizer (Optimizer or BaseOptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        monitor (str): The name of the metric to measure whether
            the performance of the model is improved.
        rule (str): One of `less`, `greater`. In `less` rule, parameters will
            be reduced when the quantity monitored has stopped
            decreasing; in `greater` rule it will be reduced when the
            quantity monitored has stopped increasing. Defaults to 'less'.
            The ``rule`` is the renaming of ``mode`` in pytorch.
        factor (float): Factor by which the parameters will be
            reduced. new_param = param * factor. Defaults to 0.1.
        patience (int): Number of epochs with no improvement after
            which parameters will be reduced. For example, if
            ``patience = 2``, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the parameters after
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
            normal operation after parameters have been reduced. Defaults to 0.
        min_value (float or list[float]): A scalar or a sequence of scalars.
            A lower bound on the parameters of each parameter group
            respectively. Defaults to 0. .
        eps (float): Minimal decay applied to parameters. If the difference
            between new and old parameters are smaller than eps, the update is
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

    .. _PyTorch ReduceLROnPlateau:
       https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
    """

    need_val_args = True

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 monitor: str = 'loss',
                 rule: str = 'less',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_rule: str = 'rel',
                 cooldown: int = 0,
                 min_value: Union[float, Sequence[float]] = 0.,
                 eps: float = 1e-8,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, BaseOptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        assert by_epoch, \
            f'Now {type(self).__name__} only support by_epoch=True'
        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))

        self.last_step = last_step

        self._global_step = 0
        self.verbose = verbose

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # This code snippet handles compatibility with the optimizer wrapper.
        # The optimizer wrapper includes an additional parameter to record the
        # base learning rate (lr) which is not affected by the paramwise_cfg.
        # By retrieving the base lr, we can obtain the actual base lr that
        # reflects the learning progress.
        if isinstance(optimizer, BaseOptimWrapper):
            raw_optimizer = optimizer.optimizer
        else:
            raw_optimizer = optimizer

        if isinstance(min_value, (list, tuple)):
            if len(min_value) != len(raw_optimizer.param_groups):
                raise ValueError('expected {} min_lrs, got {}'.format(
                    len(raw_optimizer.param_groups), len(min_value)))
            self.min_values = list(min_value)
            # Consider the `min_value` of the last param_groups
            # as the base setting. And we only add this value when
            # the optimizer is OptimWrapper.
            if isinstance(optimizer, BaseOptimWrapper) and \
                    optimizer.base_param_settings is not None:  # type: ignore
                self.min_values.append(self.min_values[-1])

        else:
            self.min_values = [min_value] * len(  # type: ignore
                optimizer.param_groups)

        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.rule_worse = None  # the worse value for the chosen mode
        self.best = None
        self.num_bad_epochs = 0
        self.eps = eps

        self.monitor = monitor
        self._init_is_better(
            rule=rule, threshold=threshold, threshold_rule=threshold_rule)
        self._reset()

        # remove call self.step() and init self._global_step = 0
        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

    def step(self, metrics=None):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule.

        Args:
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
                Defaults to None.
        """
        if metrics is None:
            # only to count self._global_step
            self._global_step += 1
            return

        if not isinstance(metrics, dict):
            raise TypeError('metrics type should be dict,'
                            f' but got type {type(metrics)}')

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1

            # convert `metric` to float, in case it's a zero-dim Tensor
            metric = metrics.get(self.monitor, None)
            if metric is not None:
                if self._is_better(metric, self.best):
                    self.best = metric
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                if self._in_cooldown():
                    self.cooldown_counter -= 1
                    self.num_bad_epochs = 0  # ignore bad epochs in cooldown

                if self.num_bad_epochs > self.patience:
                    values = self._get_value()

                    for i, data in enumerate(
                            zip(self.optimizer.param_groups, values)):
                        param_group, value = data
                        if param_group[self.param_name] - value > self.eps:
                            param_group[self.param_name] = value
                            self.print_value(self.verbose, i, value)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

            else:
                raise KeyError(f'Excepted key in {list(metrics.keys())},'
                               f' but got key {self.monitor} is not in dict')

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

    def print_value(self, is_verbose: bool, group: int, value: float) -> None:
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            step_name = 'epoch' if self.by_epoch else 'iter'
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e} '
                f'in {step_name} {self.last_step}.',
                logger='current')

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        values = [
            float(group[self.param_name]) * self.factor
            for group in self.optimizer.param_groups
        ]
        return [max(v, min_v) for v, min_v in zip(values, self.min_values)]

    def _in_cooldown(self):
        """Judge whether it is in cooldown."""
        return self.cooldown_counter > 0

    def _is_better(self, a, best):
        """Judge whether the monitor value is better."""
        if self.rule == 'less' and self.threshold_rule == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.rule == 'less' and self.threshold_rule == 'abs':
            return a < best - self.threshold

        elif self.rule == 'greater' and self.threshold_rule == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # rule == 'greater' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, rule, threshold, threshold_rule):
        """Initialize rule and its associated values."""
        if threshold < 0:
            raise ValueError(f'threshold {threshold} should be >= 0.')
        if rule not in {'less', 'greater'}:
            raise ValueError(f'mode {rule} is unknown!')
        if threshold_rule not in {'rel', 'abs'}:
            raise ValueError(f'threshold mode {threshold_rule}'
                             ' is unknown!')

        if rule == 'less':
            self.rule_worse = INF
        else:  # rule == 'greater':
            self.rule_worse = -INF

        self.rule = rule
        self.threshold = threshold
        self.threshold_rule = threshold_rule

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.rule_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
