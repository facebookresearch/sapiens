# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List, Optional

from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


def get_pred_idx(prediction: str, choices: List[str],
                 options: List[str]) -> int:  # noqa
    """Get the index (e.g. 2) from the prediction (e.g. 'C')

    Args:
        prediction (str): The prediction from the model,
            from ['A', 'B', 'C', 'D', 'E']
        choices (List(str)): The choices for the question,
            from ['A', 'B', 'C', 'D', 'E']
        options (List(str)): The options for the question,
            from ['A', 'B', 'C', 'D', 'E']

    Returns:
        int: The index of the prediction, from [0, 1, 2, 3, 4]
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


@METRICS.register_module()
class ScienceQAMetric(BaseMetric):
    """Evaluation Metric for ScienceQA.

    Args:
        options (List(str)): Options for each question. Defaults to
            ["A", "B", "C", "D", "E"].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """

    def __init__(self,
                 options: List[str] = ['A', 'B', 'C', 'D', 'E'],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.options = options

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data samples.

        data_samples should contain the following keys:
        1. pred_answer (str): The prediction from the model,
            from ['A', 'B', 'C', 'D', 'E']
        2. choices (List(str)): The choices for the question,
            from ['A', 'B', 'C', 'D', 'E']
        3. grade (int): The grade for the question, from grade1 to grade12
        4. subject (str): The subject for the question, from
            ['natural science', 'social science', 'language science']
        5. answer (str): The answer for the question, from
            ['A', 'B', 'C', 'D', 'E']
        6. hint (str): The hint for the question
        7. has_image (bool): Whether or not the question has image


        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            choices = data_sample.get('choices')
            result['prediction'] = get_pred_idx(
                data_sample.get('pred_answer'), choices, self.options)
            result['grade'] = data_sample.get('grade')
            result['subject'] = data_sample.get('subject')
            result['answer'] = data_sample.get('gt_answer')
            hint = data_sample.get('hint')
            has_image = data_sample.get('has_image', False)
            result['no_context'] = True if not has_image and len(
                hint) == 0 else False  # noqa
            result['has_text'] = True if len(hint) > 0 else False
            result['has_image'] = has_image

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = dict()

        all_acc = []
        acc_natural = []
        acc_social = []
        acc_language = []
        acc_has_text = []
        acc_has_image = []
        acc_no_context = []
        acc_grade_1_6 = []
        acc_grade_7_12 = []

        for result in results:
            correct = result['prediction'] == result['answer']
            all_acc.append(correct)
            # different subjects
            if result['subject'] == 'natural science':
                acc_natural.append(correct)
            elif result['subject'] == 'social science':
                acc_social.append(correct)
            elif result['subject'] == 'language science':
                acc_language.append(correct)

            # different context
            if result['has_text']:
                acc_has_text.append(correct)
            elif result['has_image']:
                acc_has_image.append(correct)
            elif result['no_context']:
                acc_no_context.append(correct)

            # different grade
            if result['grade'] in [
                    'grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6'
            ]:
                acc_grade_1_6.append(correct)
            elif result['grade'] in [
                    'grade7', 'grade8', 'grade9', 'grade10', 'grade11',
                    'grade12'
            ]:
                acc_grade_7_12.append(correct)

        metrics['all_acc'] = sum(all_acc) / len(all_acc)
        if len(acc_natural) > 0:
            metrics['acc_natural'] = sum(acc_natural) / len(acc_natural)
        if len(acc_social) > 0:
            metrics['acc_social'] = sum(acc_social) / len(acc_social)
        if len(acc_language) > 0:
            metrics['acc_language'] = sum(acc_language) / len(acc_language)
        if len(acc_has_text) > 0:
            metrics['acc_has_text'] = sum(acc_has_text) / len(acc_has_text)
        if len(acc_has_image) > 0:
            metrics['acc_has_image'] = sum(acc_has_image) / len(acc_has_image)
        if len(acc_no_context) > 0:
            metrics['acc_no_context'] = sum(acc_no_context) / len(
                acc_no_context)
        if len(acc_grade_1_6) > 0:
            metrics['acc_grade_1_6'] = sum(acc_grade_1_6) / len(acc_grade_1_6)
        if len(acc_grade_7_12) > 0:
            metrics['acc_grade_7_12'] = sum(acc_grade_7_12) / len(
                acc_grade_7_12)

        return metrics
