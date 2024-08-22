# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer

from mmpretrain.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class LARS(Optimizer):
    """Implements layer-wise adaptive rate scaling for SGD.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    `Large Batch Training of Convolutional Networks:
    <https://arxiv.org/abs/1708.03888>`_.

    Args:
        params (Iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Base learning rate.
        momentum (float): Momentum factor. Defaults to 0.
        weight_decay (float): Weight decay (L2 penalty). Defaults to 0.
        dampening (float): Dampening for momentum. Defaults to 0.
        eta (float): LARS coefficient. Defaults to 0.001.
        nesterov (bool): Enables Nesterov momentum. Defaults to False.
        eps (float): A small number to avoid dviding zero. Defaults to 1e-8.

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9,
        >>>                  weight_decay=1e-4, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self,
                 params: Iterable,
                 lr: float,
                 momentum: float = 0,
                 weight_decay: float = 0,
                 dampening: float = 0,
                 eta: float = 0.001,
                 nesterov: bool = False,
                 eps: float = 1e-8) -> None:
        if not isinstance(lr, float) and lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if eta < 0.0:
            raise ValueError(f'Invalid LARS coefficient value: {eta}')

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening')

        self.eps = eps
        super().__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if lars_exclude:
                    local_lr = 1.
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                    if weight_norm != 0 and grad_norm != 0:
                        # Compute local learning rate for this layer
                        local_lr = eta * weight_norm / \
                            (grad_norm + weight_decay * weight_norm + self.eps)
                    else:
                        local_lr = 1.

                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                                torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss
