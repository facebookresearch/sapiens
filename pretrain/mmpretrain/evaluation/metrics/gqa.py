# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from mmengine.evaluator import BaseMetric

from mmpretrain.evaluation.metrics.vqa import (_process_digit_article,
                                               _process_punctuation)
from mmpretrain.registry import METRICS


@METRICS.register_module()
class GQAAcc(BaseMetric):
    """GQA Acc metric.

    Compute GQA accuracy.

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
    default_prefix = 'GQA'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            gt_answer = sample.get('gt_answer')
            result = {
                'pred_answer': sample.get('pred_answer'),
                'gt_answer': gt_answer
            }

            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        acc = []
        for result in results:
            pred_answer = self._process_answer(result['pred_answer'])
            gt_answer = self._process_answer(result['gt_answer'])
            gqa_acc = 1 if pred_answer == gt_answer else 0
            acc.append(gqa_acc)

        accuracy = sum(acc) / len(acc)

        metrics = {'acc': accuracy}
        return metrics

    def _process_answer(self, answer) -> str:
        answer = _process_punctuation(answer)
        answer = _process_digit_article(answer)
        return answer
