# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.logging import print_log
from mmengine.registry import MODELS


class BaseAveragedModel(nn.Module):
    """A base class for averaging model weights.

    Weight averaging, such as SWA and EMA, is a widely used technique for
    training neural networks. This class implements the averaging process
    for a model. All subclasses must implement the `avg_func` method.
    This class creates a copy of the provided module :attr:`model`
    on the :attr:`device` and allows computing running averages of the
    parameters of the :attr:`model`.

    The code is referenced from: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py.

    Different from the `AveragedModel` in PyTorch, we use in-place operation
    to improve the parameter updating speed, which is about 5 times faster
    than the non-in-place version.

    In mmengine, we provide two ways to use the model averaging:

    1. Use the model averaging module in hook:
       We provide an :class:`mmengine.hooks.EMAHook` to apply the model
       averaging during training. Add ``custom_hooks=[dict(type='EMAHook')]``
       to the config or the runner.

    2. Use the model averaging module directly in the algorithm. Take the ema
       teacher in semi-supervise as an example:

       >>> from mmengine.model import ExponentialMovingAverage
       >>> student = ResNet(depth=50)
       >>> # use ema model as teacher
       >>> ema_teacher = ExponentialMovingAverage(student)

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: E501

    def __init__(self,
                 model: nn.Module,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__()
        self.module = deepcopy(model).requires_grad_(False)
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('steps',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.module.state_dict()
        else:
            self.avg_parameters = dict(self.module.named_parameters())

    @abstractmethod
    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Use in-place operation to compute the average of the parameters. All
        subclasses must implement this method.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1


@MODELS.register_module()
class StochasticWeightAverage(BaseAveragedModel):
    """Implements the stochastic weight averaging (SWA) of the model.

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization, UAI 2018.
    <https://arxiv.org/abs/1803.05407>`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson.
    """

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the average of the parameters using stochastic weight
        average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.add_(
            source_param - averaged_param,
            alpha=1 / float(steps // self.interval + 1))


@MODELS.register_module()
class ExponentialMovingAverage(BaseAveragedModel):
    r"""Implements the exponential moving average (EMA) of the model.

    All parameters are updated by the formula as below:

        .. math::

            Xema_{t+1} = (1 - momentum) * Xema_{t} +  momentum * X_t

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        :math:`Xema_{t+1}` is the moving average and :math:`X_t` is the
        new observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\_param = (1-momentum) * averaged\_param +
            momentum * source\_param`.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: W605

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(model, interval, device, update_buffers)
        assert 0.0 < momentum < 1.0, 'momentum must be in range (0.0, 1.0)'\
                                     f'but got {momentum}'
        if momentum > 0.5:
            print_log(
                'The value of momentum in EMA is usually a small number,'
                'which is different from the conventional notion of '
                f'momentum but got {momentum}. Please make sure the '
                f'value is correct.',
                logger='current',
                level=logging.WARNING)
        self.momentum = momentum

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using exponential
        moving average.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        averaged_param.lerp_(source_param, self.momentum)


@MODELS.register_module()
class MomentumAnnealingEMA(ExponentialMovingAverage):
    r"""Exponential moving average (EMA) with momentum annealing strategy.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
            Ema's parameter are updated with the formula
            :math:`averaged\_param = (1-momentum) * averaged\_param +
            momentum * source\_param`.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as max(momentum, gamma / (gamma + steps))
            Defaults to 100.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 100,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the linear
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = max(self.momentum,
                       self.gamma / (self.gamma + self.steps.item()))
        averaged_param.lerp_(source_param, momentum)
