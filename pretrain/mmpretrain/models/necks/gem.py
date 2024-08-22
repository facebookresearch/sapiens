# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from mmpretrain.registry import MODELS


def gem(x: Tensor, p: Parameter, eps: float = 1e-6, clamp=True) -> Tensor:
    if clamp:
        x = x.clamp(min=eps)
    return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


@MODELS.register_module()
class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        p (float): Parameter value. Defaults to 3.
        eps (float): epsilon. Defaults to 1e-6.
        clamp (bool): Use clamp before pooling. Defaults to True
        p_trainable (bool): Toggle whether Parameter p is trainable or not.
            Defaults to True.
    """

    def __init__(self, p=3., eps=1e-6, clamp=True, p_trainable=True):
        assert p >= 1, "'p' must be a value greater than 1"
        super(GeneralizedMeanPooling, self).__init__()
        self.p = Parameter(torch.ones(1) * p, requires_grad=p_trainable)
        self.eps = eps
        self.clamp = clamp
        self.p_trainable = p_trainable

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([
                gem(x, p=self.p, eps=self.eps, clamp=self.clamp)
                for x in inputs
            ])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = gem(inputs, p=self.p, eps=self.eps, clamp=self.clamp)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
