# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .cls_head import ClsHead


@MODELS.register_module()
class ConformerHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Sequence[int]): Number of channels in the input
            feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(
            self,
            num_classes: int,
            in_channels: Sequence[int],  # [conv_dim, trans_dim]
            init_cfg: dict = dict(type='TruncNormal', layer='Linear', std=.02),
            **kwargs):
        super(ConformerHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_cfg = init_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.conv_cls_head = nn.Linear(self.in_channels[0], num_classes)
        self.trans_cls_head = nn.Linear(self.in_channels[1], num_classes)

    def pre_logits(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ConformerHead``, we just obtain the
        feature of the last stage.
        """
        # The ConformerHead doesn't have other module,
        # just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """The forward process."""
        x = self.pre_logits(feats)
        # There are two outputs in the Conformer model
        assert len(x) == 2

        conv_cls_score = self.conv_cls_head(x[0])
        tran_cls_score = self.trans_cls_head(x[1])

        return conv_cls_score, tran_cls_score

    def predict(self,
                feats: Tuple[List[torch.Tensor]],
                data_samples: List[DataSample] = None) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        conv_cls_score, tran_cls_score = self(feats)
        cls_score = conv_cls_score + tran_cls_score

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_loss(self, cls_score: Tuple[torch.Tensor],
                  data_samples: List[DataSample], **kwargs) -> dict:
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = sum([
            self.loss_module(
                score, target, avg_factor=score.size(0), **kwargs)
            for score in cls_score
        ])
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(
                cls_score[0] + cls_score[1], target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses
