# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.fileio import list_from_file
from mmengine.runner import autocast
from mmengine.utils import is_seq_of

from mmpretrain.models.losses import convert_to_one_hot
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .cls_head import ClsHead


class NormProduct(nn.Linear):
    """An enhanced linear layer with k clustering centers to calculate product
    between normalized input and linear weight.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample
        k (int): The number of clustering centers. Defaults to 1.
        bias (bool): Whether there is bias. If set to ``False``, the
            layer will not learn an additive bias. Defaults to ``True``.
        feature_norm (bool): Whether to normalize the input feature.
            Defaults to ``True``.
        weight_norm (bool):Whether to normalize the weight.
            Defaults to ``True``.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 k=1,
                 bias: bool = False,
                 feature_norm: bool = True,
                 weight_norm: bool = True):

        super().__init__(in_features, out_features * k, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm
        self.out_features = out_features
        self.k = k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        cosine_all = F.linear(input, weight, self.bias)

        if self.k == 1:
            return cosine_all
        else:
            cosine_all = cosine_all.view(-1, self.out_features, self.k)
            cosine, _ = torch.max(cosine_all, dim=2)
            return cosine


@MODELS.register_module()
class ArcFaceClsHead(ClsHead):
    """ArcFace classifier head.

    A PyTorch implementation of paper `ArcFace: Additive Angular Margin Loss
    for Deep Face Recognition <https://arxiv.org/abs/1801.07698>`_ and
    `Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web
    Faces <https://link.springer.com/chapter/10.1007/978-3-030-58621-8_43>`_

    Example:
        To use ArcFace in config files.

        1. use vanilla ArcFace

        .. code:: python

            mode = dict(
                backbone = xxx,
                neck = xxxx,
                head=dict(
                    type='ArcFaceClsHead',
                    num_classes=5000,
                    in_channels=1024,
                    loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
                    init_cfg=None),
            )

        2. use SubCenterArcFace with 3 sub-centers

        .. code:: python

            mode = dict(
                backbone = xxx,
                neck = xxxx,
                head=dict(
                    type='ArcFaceClsHead',
                    num_classes=5000,
                    in_channels=1024,
                    num_subcenters=3,
                    loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
                    init_cfg=None),
            )

        3. use SubCenterArcFace With CountPowerAdaptiveMargins

        .. code:: python

            mode = dict(
                backbone = xxx,
                neck = xxxx,
                head=dict(
                    type='ArcFaceClsHead',
                    num_classes=5000,
                    in_channels=1024,
                    num_subcenters=3,
                    loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
                    init_cfg=None),
            )

            custom_hooks = [dict(type='SetAdaptiveMarginsHook')]


    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_subcenters (int): Number of subcenters. Defaults to 1.
        scale (float): Scale factor of output logit. Defaults to 64.0.
        margins (float): The penalty margin. Could be the fllowing formats:

            - float: The margin, would be same for all the categories.
            - Sequence[float]: The category-based margins list.
            - str: A '.txt' file path which contains a list. Each line
              represents the margin of a category, and the number in the
              i-th row indicates the margin of the i-th class.

            Defaults to 0.5.
        easy_margin (bool): Avoid theta + m >= PI. Defaults to False.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_subcenters: int = 1,
                 scale: float = 64.,
                 margins: Optional[Union[float, Sequence[float], str]] = 0.50,
                 easy_margin: bool = False,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = None):

        super(ArcFaceClsHead, self).__init__(init_cfg=init_cfg)
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

        assert num_subcenters >= 1 and num_classes >= 0
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.scale = scale
        self.easy_margin = easy_margin

        self.norm_product = NormProduct(in_channels, num_classes,
                                        num_subcenters)

        if isinstance(margins, float):
            margins = [margins] * num_classes
        elif isinstance(margins, str) and margins.endswith('.txt'):
            margins = [float(item) for item in list_from_file(margins)]
        else:
            assert is_seq_of(list(margins), (float, int)), (
                'the attribute `margins` in ``ArcFaceClsHead`` should be a '
                ' float, a Sequence of float, or a ".txt" file path.')

        assert len(margins) == num_classes, \
            'The length of margins must be equal with num_classes.'

        self.register_buffer(
            'margins', torch.tensor(margins).float(), persistent=False)
        # To make `phi` monotonic decreasing, refers to
        # https://github.com/deepinsight/insightface/issues/108
        sinm_m = torch.sin(math.pi - self.margins) * self.margins
        threshold = torch.cos(math.pi - self.margins)
        self.register_buffer('sinm_m', sinm_m, persistent=False)
        self.register_buffer('threshold', threshold, persistent=False)

    def set_margins(self, margins: Union[Sequence[float], float]) -> None:
        """set margins of arcface head.

        Args:
            margins (Union[Sequence[float], float]): The marigins.
        """
        if isinstance(margins, float):
            margins = [margins] * self.num_classes
        assert is_seq_of(
            list(margins), float) and (len(margins) == self.num_classes), (
                f'margins must be Sequence[Union(float, int)], get {margins}')

        self.margins = torch.tensor(
            margins, device=self.margins.device, dtype=torch.float32)
        self.sinm_m = torch.sin(self.margins) * self.margins
        self.threshold = -torch.cos(self.margins)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ArcFaceHead``, we just obtain the
        feature of the last stage.
        """
        # The ArcFaceHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def _get_logit_with_margin(self, pre_logits, target):
        """add arc margin to the cosine in target index.

        The target must be in index format.
        """
        assert target.dim() == 1 or (
            target.dim() == 2 and target.shape[1] == 1), \
            'The target must be in index format.'
        cosine = self.norm_product(pre_logits)
        phi = torch.cos(torch.acos(cosine) + self.margins)

        if self.easy_margin:
            # when cosine>0, choose phi
            # when cosine<=0, choose cosine
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # when cos>th, choose phi
            # when cos<=th, choose cosine-mm
            phi = torch.where(cosine > self.threshold, phi,
                              cosine - self.sinm_m)

        target = convert_to_one_hot(target, self.num_classes)
        output = target * phi + (1 - target) * cosine
        return output

    def forward(self,
                feats: Tuple[torch.Tensor],
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The forward process."""
        # Disable AMP
        with autocast(enabled=False):
            pre_logits = self.pre_logits(feats)

            if target is None:
                # when eval, logit is the cosine between W and pre_logits;
                # cos(theta_yj) = (x/||x||) * (W/||W||)
                logit = self.norm_product(pre_logits)
            else:
                # when training, add a margin to the pre_logits where target is
                # True, then logit is the cosine between W and new pre_logits
                logit = self._get_logit_with_margin(pre_logits, target)

        return self.scale * logit

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Unpack data samples and pack targets
        label_target = torch.cat([i.gt_label for i in data_samples])
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = label_target

        # the index format target would be used
        cls_score = self(feats, label_target)

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses
