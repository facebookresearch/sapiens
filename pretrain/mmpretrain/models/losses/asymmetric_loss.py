# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .utils import convert_to_one_hot, weight_reduce_loss


def asymmetric_loss(pred,
                    target,
                    weight=None,
                    gamma_pos=1.0,
                    gamma_neg=4.0,
                    clip=0.05,
                    reduction='mean',
                    avg_factor=None,
                    use_sigmoid=True,
                    eps=1e-8):
    r"""asymmetric loss.

    Please refer to the `paper <https://arxiv.org/abs/2009.14119>`__ for
    details.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The ground truth label of the prediction with
            shape (N, \*).
        weight (torch.Tensor, optional): Sample-wise loss weight with shape
            (N, ). Defaults to None.
        gamma_pos (float): positive focusing parameter. Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We usually set
            gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.

    Returns:
        torch.Tensor: Loss.
    """
    assert pred.shape == \
        target.shape, 'pred and target should be in the same shape.'

    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = nn.functional.softmax(pred, dim=-1)

    target = target.type_as(pred)

    if clip and clip > 0:
        pt = (1 - pred_sigmoid +
              clip).clamp(max=1) * (1 - target) + pred_sigmoid * target
    else:
        pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    asymmetric_weight = (1 - pt).pow(gamma_pos * target + gamma_neg *
                                     (1 - target))
    loss = -torch.log(pt.clamp(min=eps)) * asymmetric_weight
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class AsymmetricLoss(nn.Module):
    """asymmetric loss.

    Args:
        gamma_pos (float): positive focusing parameter.
            Defaults to 0.0.
        gamma_neg (float): Negative focusing parameter. We
            usually set gamma_neg > gamma_pos. Defaults to 4.0.
        clip (float, optional): Probability margin. Defaults to 0.05.
        reduction (str): The method used to reduce the loss into
            a scalar.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        use_sigmoid (bool): Whether the prediction uses sigmoid instead
            of softmax. Defaults to True.
        eps (float): The minimum value of the argument of logarithm. Defaults
            to 1e-8.
    """

    def __init__(self,
                 gamma_pos=0.0,
                 gamma_neg=4.0,
                 clip=0.05,
                 reduction='mean',
                 loss_weight=1.0,
                 use_sigmoid=True,
                 eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        r"""asymmetric loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            target (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*), N or (N,1).
            weight (torch.Tensor, optional): Sample-wise loss weight with shape
                (N, \*). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss into a scalar. Options are "none", "mean" and "sum".
                Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            target = convert_to_one_hot(target.view(-1, 1), pred.shape[-1])
        loss_cls = self.loss_weight * asymmetric_loss(
            pred,
            target,
            weight,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            clip=self.clip,
            reduction=reduction,
            avg_factor=avg_factor,
            use_sigmoid=self.use_sigmoid,
            eps=self.eps)
        return loss_cls
