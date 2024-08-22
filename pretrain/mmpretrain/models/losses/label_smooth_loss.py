# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cross_entropy_loss import CrossEntropyLoss
from .utils import convert_to_one_hot


@MODELS.register_module()
class LabelSmoothLoss(nn.Module):
    r"""Initializer for the label smoothed cross entropy loss.

    Refers to `Rethinking the Inception Architecture for Computer Vision
    <https://arxiv.org/abs/1512.00567>`_

    This decreases gap between output scores and encourages generalization.
    Labels provided to forward can be one-hot like vectors (NxC) or class
    indices (Nx1).
    And this accepts linear combination of one-hot like labels from mixup or
    cutmix except multi-label task.

    Args:
        label_smooth_val (float): The degree of label smoothing.
        num_classes (int, optional): Number of classes. Defaults to None.
        mode (str): Refers to notes, Options are 'original', 'classy_vision',
            'multi_label'. Defaults to 'original'.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid of
            softmax. Defaults to None, which means to use sigmoid in
            "multi_label" mode and not use in other modes.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.

    Notes:
        - if the mode is **"original"**, this will use the same label smooth
          method as the original paper as:

          .. math::
              (1-\epsilon)\delta_{k, y} + \frac{\epsilon}{K}

          where :math:`\epsilon` is the ``label_smooth_val``, :math:`K` is the
          ``num_classes`` and :math:`\delta_{k, y}` is Dirac delta, which
          equals 1 for :math:`k=y` and 0 otherwise.

        - if the mode is **"classy_vision"**, this will use the same label
          smooth method as the facebookresearch/ClassyVision repo as:

          .. math::
              \frac{\delta_{k, y} + \epsilon/K}{1+\epsilon}

        - if the mode is **"multi_label"**, this will accept labels from
          multi-label task and smoothing them as:

          .. math::
              (1-2\epsilon)\delta_{k, y} + \epsilon
    """

    def __init__(self,
                 label_smooth_val,
                 num_classes=None,
                 use_sigmoid=None,
                 mode='original',
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight

        assert (isinstance(label_smooth_val, float)
                and 0 <= label_smooth_val < 1), \
            f'LabelSmoothLoss accepts a float label_smooth_val ' \
            f'over [0, 1), but gets {label_smooth_val}'
        self.label_smooth_val = label_smooth_val

        accept_reduction = {'none', 'mean', 'sum'}
        assert reduction in accept_reduction, \
            f'LabelSmoothLoss supports reduction {accept_reduction}, ' \
            f'but gets {mode}.'
        self.reduction = reduction

        accept_mode = {'original', 'classy_vision', 'multi_label'}
        assert mode in accept_mode, \
            f'LabelSmoothLoss supports mode {accept_mode}, but gets {mode}.'
        self.mode = mode

        self._eps = label_smooth_val
        if mode == 'classy_vision':
            self._eps = label_smooth_val / (1 + label_smooth_val)

        if mode == 'multi_label':
            if not use_sigmoid:
                from mmengine.logging import MMLogger
                MMLogger.get_current_instance().warning(
                    'For multi-label tasks, please set `use_sigmoid=True` '
                    'to use binary cross entropy.')
            self.smooth_label = self.multilabel_smooth_label
            use_sigmoid = True if use_sigmoid is None else use_sigmoid
        else:
            self.smooth_label = self.original_smooth_label
            use_sigmoid = False if use_sigmoid is None else use_sigmoid

        self.ce = CrossEntropyLoss(
            use_sigmoid=use_sigmoid,
            use_soft=not use_sigmoid,
            reduction=reduction,
            class_weight=class_weight,
            pos_weight=pos_weight)

    def generate_one_hot_like_label(self, label):
        """This function takes one-hot or index label vectors and computes one-
        hot like label vectors (float)"""
        # check if targets are inputted as class integers
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def original_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = one_hot_like_label * (1 - self._eps)
        smooth_label += self._eps / self.num_classes
        return smooth_label

    def multilabel_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = torch.full_like(one_hot_like_label, self._eps)
        smooth_label.masked_fill_(one_hot_like_label > 0, 1 - self._eps)
        return smooth_label

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        r"""Label smooth loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, \*).
            label (torch.Tensor): The ground truth label of the prediction
                with shape (N, \*).
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
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        assert one_hot_like_label.shape == cls_score.shape, \
            f'LabelSmoothLoss requires output and target ' \
            f'to be same shape, but got output.shape: {cls_score.shape} ' \
            f'and target.shape: {one_hot_like_label.shape}'

        smoothed_label = self.smooth_label(one_hot_like_label)
        return self.loss_weight * self.ce.forward(
            cls_score,
            smoothed_label,
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
            **kwargs)
