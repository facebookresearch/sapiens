# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torchvision.ops.boxes as boxes
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


def aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = boxes.box_area(boxes1)
    area2 = boxes.box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (B, 2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (B, 2)

    wh = boxes._upcast(rb - lt).clamp(min=0)  # (B, 2)
    inter = wh[:, 0] * wh[:, 1]  # (B, )

    union = area1 + area2 - inter
    iou = inter / union
    return iou


@METRICS.register_module()
class VisualGroundingMetric(BaseMetric):
    """Visual Grounding evaluator.

    Calculate the box mIOU and box grounding accuracy for visual grounding
    model.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """
    default_prefix = 'visual-grounding'

    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for preds in data_samples:

            pred_box = preds['pred_bboxes'].squeeze()
            box_gt = torch.Tensor(preds['gt_bboxes']).squeeze()

            result = {
                'box': pred_box.to('cpu').squeeze(),
                'box_target': box_gt.squeeze(),
            }

            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        pred_boxes = torch.stack([each['box'] for each in results])
        gt_boxes = torch.stack([each['box_target'] for each in results])
        iou = aligned_box_iou(pred_boxes, gt_boxes)
        accu_num = torch.sum(iou >= 0.5)

        miou = torch.mean(iou)
        acc = accu_num / len(gt_boxes)
        coco_val = {'miou': miou, 'acc': acc}
        return coco_val
