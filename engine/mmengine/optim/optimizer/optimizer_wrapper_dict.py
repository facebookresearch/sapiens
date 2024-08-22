# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .optimizer_wrapper import OptimWrapper


class OptimWrapperDict(OptimWrapper):
    """A dictionary container of :obj:`OptimWrapper`.

    If runner is training with multiple optimizers, all optimizer wrappers
    should be managed by :obj:`OptimWrapperDict` which is built by
    ``CustomOptimWrapperConstructor``. ``OptimWrapperDict`` will load and save
    the state dictionary of all optimizer wrappers.

    Consider the semantic ambiguity of calling :meth:``update_params``,
    :meth:`backward` of all optimizer wrappers, ``OptimWrapperDict`` will not
    implement these methods.

    Examples:
        >>> import torch.nn as nn
        >>> from torch.optim import SGD
        >>> from mmengine.optim import OptimWrapperDict, OptimWrapper
        >>> model1 = nn.Linear(1, 1)
        >>> model2 = nn.Linear(1, 1)
        >>> optim_wrapper1 = OptimWrapper(SGD(model1.parameters(), lr=0.1))
        >>> optim_wrapper2 = OptimWrapper(SGD(model2.parameters(), lr=0.1))
        >>> optim_wrapper_dict = OptimWrapperDict(model1=optim_wrapper1,
        >>>                                       model2=optim_wrapper2)

    Note:
        The optimizer wrapper contained in ``OptimWrapperDict`` can be accessed
        in the same way as `dict`.

    Args:
        **optim_wrappers: A dictionary of ``OptimWrapper`` instance.
    """

    def __init__(self, **optim_wrapper_dict: OptimWrapper):
        for key, value in optim_wrapper_dict.items():
            assert isinstance(value, OptimWrapper), (
                '`OptimWrapperDict` only accept OptimWrapper instance, '
                f'but got {key}: {type(value)}')
        self.optim_wrappers = optim_wrapper_dict

    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update all optimizer wrappers would lead to a duplicate backward
        errors, and OptimWrapperDict does not know which optimizer wrapper
        should be updated.

        Therefore, this method is not implemented. The optimizer wrapper of
        OptimWrapperDict should be accessed and call its `update_params`.
        """
        raise NotImplementedError('`update_params` should be called by each '
                                  'optimizer separately`')

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Since OptimWrapperDict doesn't know which optimizer wrapper's
        backward method should be called (``loss_scaler`` maybe different in
        different :obj:AmpOptimWrapper), this method is not implemented.

        The optimizer wrapper of OptimWrapperDict should be accessed and call
        its `backward`.
        """
        raise NotImplementedError('`backward` should be called by each '
                                  'optimizer separately`')

    def step(self, **kwargs) -> None:
        """Since the backward method is not implemented, the step should not be
        implemented either."""
        raise NotImplementedError('`step` should be called by each '
                                  'optimizer separately`')

    def zero_grad(self, **kwargs) -> None:
        """Set the gradients of all optimizer wrappers to zero."""
        for optim_wrapper in self.optim_wrappers.values():
            optim_wrapper.zero_grad()

    @contextmanager
    def optim_context(self, model: nn.Module):
        """``optim_context`` should be called by each optimizer separately."""
        raise NotImplementedError(
            '`optim_context` should be called by each optimizer separately')

    def initialize_count_status(self, model: nn.Module, cur_iter,
                                max_iters) -> None:
        """Do nothing but provide unified interface for :obj:`OptimWrapper`

        Since ``OptimWrapperDict`` does not know the correspondence between
        model and optimizer wrapper. ``initialize_iter_status`` will do nothing
        and each optimizer wrapper should call ``initialize_iter_status``
        separately.
        """
        return

    @property
    def param_groups(self):
        """Returns the parameter groups of each OptimWrapper."""
        param_groups = dict()
        for key, value in self.optim_wrappers.items():
            param_groups[key] = value.param_groups
        return param_groups

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of all optimizers.

        Returns:
            Dict[str, List[float]]: Learning rate of all optimizers.
        """
        lr_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            inner_lr_dict = optim_wrapper.get_lr()
            if 'base_lr' in inner_lr_dict:
                lr_dict[f'{name}.base_lr'] = inner_lr_dict['base_lr']
            lr_dict[f'{name}.lr'] = inner_lr_dict['lr']
        return lr_dict

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of all optimizers.

        Returns:
            Dict[str, List[float]]: momentum of all optimizers.
        """
        momentum_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            momentum_dict[f'{name}.momentum'] = optim_wrapper.get_momentum(
            )['momentum']
        return momentum_dict

    def state_dict(self) -> dict:
        """Get the state dictionary of all optimizer wrappers.

        Returns:
            dict: Each key-value pair in the dictionary represents the name
            and state dictionary of corresponding :obj:`OptimWrapper`.
        """
        state_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            state_dict[name] = optim_wrapper.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dictionary from the ``state_dict``.

        Args:
            state_dict (dict): Each key-value pair in `state_dict` represents
                the name and the state dictionary of corresponding
                :obj:`OptimWrapper`.
        """
        for name, _state_dict in state_dict.items():
            assert name in self.optim_wrappers, (
                f'Mismatched `state_dict`! cannot found {name} in '
                'OptimWrapperDict')
            self.optim_wrappers[name].load_state_dict(_state_dict)

    def items(self) -> Iterator[Tuple[str, OptimWrapper]]:
        """A generator to get the name and corresponding
        :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.items()

    def values(self) -> Iterator[OptimWrapper]:
        """A generator to get :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.values()

    def keys(self) -> Iterator[str]:
        """A generator to get the name of :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.keys()

    def __getitem__(self, key: str) -> OptimWrapper:
        assert key in self.optim_wrappers, (
            f'Cannot find {key} in OptimWrapperDict, please check '
            'your optimizer constructor.')
        return self.optim_wrappers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.optim_wrappers

    def __len__(self) -> int:
        return len(self.optim_wrappers)

    def __repr__(self) -> str:
        desc = ''
        for name, optim_wrapper in self.optim_wrappers.items():
            desc += f'name: {name}\n'
            desc += repr(optim_wrapper)
        return desc
