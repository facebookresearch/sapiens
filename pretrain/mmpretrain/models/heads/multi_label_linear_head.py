# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .multi_label_cls_head import MultiLabelClsHead


@MODELS.register_module()
class MultiLabelLinearClsHead(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        loss (dict): Config of classification loss. Defaults to
            dict(type='CrossEntropyLoss', use_sigmoid=True).
        thr (float, optional): Predictions with scores under the thresholds
            are considered as negative. Defaults to None.
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive. Defaults to None.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).

    Notes:
        If both ``thr`` and ``topk`` are set, use ``thr` to determine
        positive predictions. If neither is set, use ``thr=0.5`` as
        default.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss: Dict = dict(type='CrossEntropyLoss', use_sigmoid=True),
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelLinearClsHead, self).__init__(
            loss=loss, thr=thr, topk=topk, init_cfg=init_cfg)

        assert num_classes > 0, f'num_classes ({num_classes}) must be a ' \
            'positive integer.'

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``MultiLabelLinearClsHead``, we just
        obtain the feature of the last stage.
        """
        # The obtain the MultiLabelLinearClsHead doesn't have other module,
        # just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
