# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

from mmpretrain.registry import METRICS
from mmpretrain.structures import label_to_onehot
from .multi_label import AveragePrecision, MultiLabelMetric


class VOCMetricMixin:
    """A mixin class for VOC dataset metrics, VOC annotations have extra
    `difficult` attribute for each object, therefore, extra option is needed
    for calculating VOC metrics.

    Args:
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive in one-hot ground truth for evaluation. If it
            set to True, map difficult gt labels to positive ones(1), If it
            set to False, map difficult gt labels to negative ones(0).
            Defaults to None, the difficult labels will be set to '-1'.
    """

    def __init__(self,
                 *arg,
                 difficult_as_positive: Optional[bool] = None,
                 **kwarg):
        self.difficult_as_positive = difficult_as_positive
        super().__init__(*arg, **kwarg)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            gt_label = data_sample['gt_label']
            gt_label_difficult = data_sample['gt_label_difficult']

            result['pred_score'] = data_sample['pred_score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'gt_score' in data_sample:
                result['gt_score'] = data_sample['gt_score'].clone()
            else:
                result['gt_score'] = label_to_onehot(gt_label, num_classes)

            # VOC annotation labels all the objects in a single image
            # therefore, some categories are appeared both in
            # difficult objects and non-difficult objects.
            # Here we reckon those labels which are only exists in difficult
            # objects as difficult labels.
            difficult_label = set(gt_label_difficult) - (
                set(gt_label_difficult) & set(gt_label.tolist()))

            # set difficult label for better eval
            if self.difficult_as_positive is None:
                result['gt_score'][[*difficult_label]] = -1
            elif self.difficult_as_positive:
                result['gt_score'][[*difficult_label]] = 1

            # Save the result to `self.results`.
            self.results.append(result)


@METRICS.register_module()
class VOCMultiLabelMetric(VOCMetricMixin, MultiLabelMetric):
    """A collection of metrics for multi-label multi-class classification task
    based on confusion matrix for VOC dataset.

    It includes precision, recall, f1-score and support.

    Args:
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive in one-hot ground truth for evaluation. If it
            set to True, map difficult gt labels to positive ones(1), If it
            set to False, map difficult gt labels to negative ones(0).
            Defaults to None, the difficult labels will be set to '-1'.
        **kwarg: Refers to `MultiLabelMetric` for detailed docstrings.
    """


@METRICS.register_module()
class VOCAveragePrecision(VOCMetricMixin, AveragePrecision):
    """Calculate the average precision with respect of classes for VOC dataset.

    Args:
        difficult_as_postive (Optional[bool]): Whether to map the difficult
            labels as positive in one-hot ground truth for evaluation. If it
            set to True, map difficult gt labels to positive ones(1), If it
            set to False, map difficult gt labels to negative ones(0).
            Defaults to None, the difficult labels will be set to '-1'.
        **kwarg: Refers to `AveragePrecision` for detailed docstrings.
    """
