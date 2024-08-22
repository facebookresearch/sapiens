# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import Dict, List

import torch


class BaseOptimWrapper(metaclass=ABCMeta):

    def __init__(self, optimizer):
        self.optimizer = optimizer

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

    @abstractmethod
    def update_params(self, *args, **kwargs):
        """Update parameters in :attr:`optimizer`."""

    @abstractmethod
    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation."""

    @abstractmethod
    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``."""

    @abstractmethod
    def step(self, **kwargs):
        """Call the step method of optimizer."""

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``."""
        state_dict = self.optimizer.state_dict()
        if self.base_param_settings is not None:
            state_dict['base_param_settings'] = self.base_param_settings
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Provide unified ``load_state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be loaded when training with ``torch.cuda.amp``.

        Args:
            state_dict (dict): The state dictionary of :attr:`optimizer`.
        """
        base_param_settings = state_dict.pop('base_param_settings', None)

        if base_param_settings is not None:
            self.base_param_settings = base_param_settings

        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        if self.base_param_settings is not None:
            return self.optimizer.param_groups + [self.base_param_settings]
        else:
            return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """A wrapper of ``Optimizer.defaults``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.defaults

    def get_lr(self):
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]:
            param_groups learning rate of the optimizer.
        """
        res = {}
        if self.base_param_settings is not None:
            res['base_lr'] = [self.base_param_settings['lr']]

        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]

        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)
