# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# migrate from mmdetection with modifications
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from .utils import weight_reduce_loss


def seesaw_ce_loss(cls_score,
                   labels,
                   weight,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   eps,
                   reduction='mean',
                   avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.

    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index
                                                               )  # M_{ij}
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class SeesawLoss(nn.Module):
    """Implementation of seesaw loss.

    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>`_

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid of softmax.
             Only False is supported. Defaults to False.
        p (float): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int): The number of classes.
             Defaults to 1000 for the ImageNet dataset.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor, default to 1e-2.
        reduction (str): The method that reduces the loss to a scalar.
             Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1000,
                 eps=1e-2,
                 reduction='mean',
                 loss_weight=1.0):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid, '`use_sigmoid` is not supported'
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        self.register_buffer('cum_samples',
                             torch.zeros(self.num_classes, dtype=torch.float))

    def forward(self,
                cls_score,
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            f'The `reduction_override` should be one of (None, "none", ' \
            f'"mean", "sum"), but get "{reduction_override}".'
        assert cls_score.size(0) == labels.view(-1).size(0), \
            f'Expected `labels` shape [{cls_score.size(0)}], ' \
            f'but got {list(labels.size())}'
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes, \
            f'The channel number of output ({cls_score.size(-1)}) does ' \
            f'not match the `num_classes` of seesaw loss ({self.num_classes}).'

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if weight is not None:
            weight = weight.float()
        else:
            weight = labels.new_ones(labels.size(), dtype=torch.float)

        # calculate loss_cls_classes
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, labels, weight, self.cum_samples, self.num_classes,
            self.p, self.q, self.eps, reduction, avg_factor)

        return loss_cls
