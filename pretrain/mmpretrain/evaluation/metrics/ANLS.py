# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


@METRICS.register_module()
class ANLS(BaseMetric):
    """ANLS metric.

    Compute the Average Normalized Levenshtein Similarity(ANLS).

    Args:
        threshold (float): ANLS threshold used for determining if the answer
            has been correctly selected but not properly recognized,
            or on the contrary, the output is a wrong text selected from the
            options and given as an answer.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """
    default_prefix = 'ANLS'

    def __init__(self,
                 threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold = threshold

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
        total_score = 0.
        for result in results:
            sample_score_list = []
            pred = ' '.join(result['pred_answer'].strip().lower().split())
            for gt in result['gt_answer']:
                gt = ' '.join(gt.strip().lower().split())
                dist = levenshtein_distance(gt, pred)
                length = max(
                    len(gt.upper()), len(result['pred_answer'].upper()))
                sample_score_list.append(0.0 if length == 0 else float(dist) /
                                         float(length))

            per_sample_score = 1. - min(sample_score_list)
            if per_sample_score < self.threshold:
                per_sample_score = 0.

            total_score += per_sample_score

        total_score = total_score / len(results)
        return {'ANLS': total_score}


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1],
                                           distances_[-1])))
        distances = distances_
    return distances[-1]
