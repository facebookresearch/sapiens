# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import cos, pi
from typing import Optional

import torch
import torch.nn as nn
from mmengine.logging import MessageHub
from mmengine.model import ExponentialMovingAverage

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CosineEMA(ExponentialMovingAverage):
    r"""CosineEMA is implemented for updating momentum parameter, used in BYOL,
    MoCoV3, etc.

    All parameters are updated by the formula as below:

    .. math::

        X'_{t+1} = (1 - m) * X'_t + m * X_t

    Where :math:`m` the the momentum parameter. And it's updated with cosine
    annealing, including momentum adjustment following:

    .. math::
        m = m_{end} + (m_{end} - m_{start}) * (\cos\frac{k\pi}{K} + 1) / 2

    where :math:`k` is the current step, :math:`K` is the total steps.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically,
        :math:`X'_{t}` is the moving average and :math:`X_t` is the new
        observed value. The value of momentum is usually a small number,
        allowing observed values to slowly update the ema parameters. See also
        :external:py:class:`torch.nn.BatchNorm2d`.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The start momentum value. Defaults to 0.004.
        end_momentum (float): The end momentum value for cosine annealing.
            Defaults to 0.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.004,
                 end_momentum: float = 0.,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False) -> None:
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        self.end_momentum = end_momentum

    def avg_func(self, averaged_param: torch.Tensor,
                 source_param: torch.Tensor, steps: int) -> None:
        """Compute the moving average of the parameters using the cosine
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.

        Returns:
            Tensor: The averaged parameters.
        """
        message_hub = MessageHub.get_current_instance()
        max_iters = message_hub.get_info('max_iters')
        cosine_annealing = (cos(pi * steps / float(max_iters)) + 1) / 2
        momentum = self.end_momentum - (self.end_momentum -
                                        self.momentum) * cosine_annealing
        averaged_param.mul_(1 - momentum).add_(source_param, alpha=momentum)
