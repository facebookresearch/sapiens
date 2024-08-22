# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from mmengine.logging import MessageHub, print_log
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.utils.dl_utils import has_batch_norm
from .base import BaseOptimWrapper


@OPTIM_WRAPPERS.register_module()
class OptimWrapper(BaseOptimWrapper):
    """Optimizer wrapper provides a common interface for updating parameters.

    Optimizer wrapper provides a unified interface for single precision
    training and automatic mixed precision training with different hardware.
    OptimWrapper encapsulates optimizer to provide simplified interfaces
    for commonly used training techniques such as gradient accumulative and
    grad clips. ``OptimWrapper`` implements the basic logic of gradient
    accumulation and gradient clipping based on ``torch.optim.Optimizer``.
    The subclasses only need to override some methods to implement the mixed
    precision training. See more information in :class:`AmpOptimWrapper`.

    Args:
        optimizer (Optimizer): Optimizer used to update model parameters.
        accumulative_counts (int): The number of iterations to accumulate
            gradients. The parameters will be updated per
            ``accumulative_counts``.
        clip_grad (dict, optional): If ``clip_grad`` is not None, it will be
            the arguments of :func:`torch.nn.utils.clip_grad_norm_` or
            :func:`torch.nn.utils.clip_grad_value_`. ``clip_grad`` should be a
            dict, and the keys could be set as follows:

            If the key ``type`` is not set, or ``type`` is "norm",
            the accepted keys are as follows:

            - max_norm (float or int): Max norm of the gradients.
            - norm_type (float or int): Type of the used p-norm. Can be
              ``'inf'`` for infinity norm.
            - error_if_nonfinite (bool): If True, an error is thrown if
              the total norm of the gradients from :attr:`parameters` is
              ``nan``, ``inf``, or ``-inf``. Defaults to False (will switch
              to True in the future)

            If the key ``type`` is set to "value", the accepted keys are as
            follows:

            - clip_value (float or int): maximum allowed value of the
              gradients. The gradients are clipped in the range
              ``(-clip_value, +clip_value)``.

    Note:
        If ``accumulative_counts`` is larger than 1, perform
        :meth:`update_params` under the context of  ``optim_context``
        could avoid unnecessary gradient synchronization.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.

    Note:
        The subclass should ensure that once :meth:`update_params` is called,
        ``_inner_count += 1`` is automatically performed.

    Examples:
        >>> # Config sample of OptimWrapper and enable clipping gradient by
        >>> # norm.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=1,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> # Config sample of OptimWrapper and enable clipping gradient by
        >>> # value.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=1,
        >>>     clip_grad=dict(type='value', clip_value=0.2))
        >>> # Use OptimWrapper to update model.
        >>> import torch.nn as nn
        >>> import torch
        >>> from torch.optim import SGD
        >>> from torch.utils.data import DataLoader
        >>> from mmengine.optim import OptimWrapper
        >>>
        >>> model = nn.Linear(1, 1)
        >>> dataset = torch.randn(10, 1, 1)
        >>> dataloader = DataLoader(dataset)
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>>
        >>> for data in dataloader:
        >>>     loss = model(data)
        >>>     optim_wrapper.update_params(loss)
        >>> # Enable gradient accumulation
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=3,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> ddp_model = DistributedDataParallel(model)
        >>> optimizer = SGD(ddp_model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>> optim_wrapper.initialize_count_status(0, len(dataloader))
        >>> # If model is a subclass instance of DistributedDataParallel,
        >>> # `optim_context` context manager can avoid unnecessary gradient
        >>> #  synchronize.
        >>> for iter, data in enumerate(dataloader):
        >>>     with optim_wrapper.optim_context(ddp_model):
        >>>         loss = model(data)
        >>>     optim_wrapper.update_params(loss)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        assert accumulative_counts > 0, (
            '_accumulative_counts at least greater than or equal to 1')
        self._accumulative_counts = accumulative_counts
        self.optimizer = optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                'If `clip_grad` is not None, it should be a `dict` '
                'which is the arguments of `torch.nn.utils.clip_grad_norm_` '
                'or clip_grad_value_`.')
            clip_type = clip_grad.pop('type', 'norm') ## if type not found then defaults to 'norm'
            if clip_type == 'norm':
                self.clip_func = torch.nn.utils.clip_grad_norm_
                self.grad_name = 'grad_norm'
            elif clip_type == 'value':
                self.clip_func = torch.nn.utils.clip_grad_value_
                self.grad_name = 'grad_value'
            else:
                raise ValueError('type of clip_grad should be "norm" or '
                                 f'"value" but got {clip_type}')
            assert clip_grad, ('`clip_grad` should contain other arguments '
                               'besides `type`. The arguments should match '
                               'with the `torch.nn.utils.clip_grad_norm_` or '
                               'clip_grad_value_`')
        self.clip_grad_kwargs = clip_grad
        # Used to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        # `_max_counts` means the total number of parameter updates.  It
        # ensures that the gradient of the last few iterations will not be
        # lost when the `_max_counts` is not divisible by
        # `accumulative_counts`.
        self._max_counts = -1
        # The `_remainder_iter` is used for calculating loss factor at the
        # last few iterations. If `_max_counts` has not been initialized,
        # the loss factor will always be the same as `_accumulative_counts`.
        self._remainder_counts = -1

        # The Following code is used to initialize `base_param_settings`.
        # `base_param_settings` is used to store the parameters that are not
        # updated by the optimizer.
        # The `base_param_settings` used for tracking the base learning in the
        # optimizer. If the optimizer has multiple parameter groups, this
        # params will not be scaled by the loss factor.
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                'params': torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None  # type: ignore

    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Note:
            If subclasses inherit from ``OptimWrapper`` override
            ``backward``, ``_inner_count +=1`` must be implemented.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`.
        """
        loss.backward(**kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Provide unified ``zero_grad`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.zero_grad`.
        """
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        ##-------------zero out nan--------------
        params = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        for p in params:
            if hasattr(p, "grad") and p.requires_grad and p.grad is not None:
                p.grad.data[torch.isnan(p.grad.data)] = 0
                p.grad.data[torch.isinf(p.grad.data)] = 0

        ##----------------------------------------
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            # `torch.nn.utils.clip_grad_value_` will return None.

            if grad is not None:
                self.message_hub.update_scalar(f'train/{self.grad_name}',
                                               float(grad))

    def initialize_count_status(self, model: nn.Module, init_counts: int,
                                max_counts: int) -> None:
        """Initialize gradient accumulation related attributes.

        ``OptimWrapper`` can be used without calling
        ``initialize_iter_status``. However, Consider the case of  ``len(
        dataloader) == 10``, and the ``accumulative_iter == 3``. Since 10 is
        not divisible by 3, the last iteration will not trigger
        ``optimizer.step()``, resulting in one less parameter updating.

        Args:
            model (nn.Module): Training model
            init_counts (int): The initial value of the inner count.
            max_counts (int): The maximum value of the inner count.
        """
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            print_log(
                'Resumed iteration number is not divisible by '
                '`_accumulative_counts` in `GradientCumulativeOptimizerHook`, '
                'which means the gradient of some iterations is lost and the '
                'result may be influenced slightly.',
                logger='current',
                level=logging.WARNING)

        if has_batch_norm(model) and self._accumulative_counts > 1:
            print_log(
                'Gradient accumulative may slightly decrease '
                'performance because the model has BatchNorm layers.',
                logger='current',
                level=logging.WARNING)
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return ((self._inner_count + 1) % self._accumulative_counts == 0
                or (self._inner_count + 1) == self._max_counts)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Get scaled loss according to ``_accumulative_counts``,
        ``_inner_count`` and max_counts.

        Args:
            loss (torch.Tensor): Original loss calculated by model.

        Returns:
            loss (torch.Tensor): Scaled loss.
        """
        if self._accumulative_counts == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            # if `self._accumulative_counts > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self._accumulative_counts`. However, `self._max_counts` may not
            # be divisible by `self._accumulative_counts`, so the
            # `loss_scale` for the last few iterations needs to be
            # recalculated.
            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when initialize_iter_status called with an '
                'error `init_counts` or `max_counts`')

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (f'Type: {type(self).__name__}\n'
                        f'_accumulative_counts: {self._accumulative_counts}\n'
                        'optimizer: \n')
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str
