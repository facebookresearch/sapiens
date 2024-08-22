# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Sequence

from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


@METRICS.register_module()
class MultiTasksMetric(BaseMetric):
    """Metrics for MultiTask
    Args:
        task_metrics(dict): a dictionary in the keys are the names of the tasks
            and the values is a list of the metric corresponds to this task
    Examples:
        >>> import torch
        >>> from mmpretrain.evaluation import MultiTasksMetric
        # -------------------- The Basic Usage --------------------
        >>>task_metrics = {
            'task0': [dict(type='Accuracy', topk=(1, ))],
            'task1': [dict(type='Accuracy', topk=(1, 3))]
        }
        >>>pred = [{
            'pred_task': {
                'task0': torch.tensor([0.7, 0.0, 0.3]),
                'task1': torch.tensor([0.5, 0.2, 0.3])
            },
            'gt_task': {
                'task0':  torch.tensor(0),
                'task1':  torch.tensor(2)
            }
        }, {
            'pred_task': {
                'task0': torch.tensor([0.0, 0.0, 1.0]),
                'task1': torch.tensor([0.0, 0.0, 1.0])
            },
            'gt_task': {
                'task0':  torch.tensor(2),
                'task1':  torch.tensor(2)
            }
        }]
        >>>metric = MultiTasksMetric(task_metrics)
        >>>metric.process(None, pred)
        >>>results = metric.evaluate(2)
        results = {
            'task0_accuracy/top1': 100.0,
            'task1_accuracy/top1': 50.0,
            'task1_accuracy/top3': 100.0
        }
    """

    def __init__(self,
                 task_metrics: Dict,
                 collect_device: str = 'cpu') -> None:
        self.task_metrics = task_metrics
        super().__init__(collect_device=collect_device)

        self._metrics = {}
        for task_name in self.task_metrics.keys():
            self._metrics[task_name] = []
            for metric in self.task_metrics[task_name]:
                self._metrics[task_name].append(METRICS.build(metric))

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for task_name in self.task_metrics.keys():
            filtered_data_samples = []
            for data_sample in data_samples:
                eval_mask = data_sample[task_name]['eval_mask']
                if eval_mask:
                    filtered_data_samples.append(data_sample[task_name])
            for metric in self._metrics[task_name]:
                metric.process(data_batch, filtered_data_samples)

    def compute_metrics(self, results: list) -> dict:
        raise NotImplementedError(
            'compute metrics should not be used here directly')

    def evaluate(self, size):
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are
            "{task_name}_{metric_name}" , and the values
            are corresponding results.
        """
        metrics = {}
        for task_name in self._metrics:
            for metric in self._metrics[task_name]:
                name = metric.__class__.__name__
                if name == 'MultiTasksMetric' or metric.results:
                    results = metric.evaluate(size)
                else:
                    results = {metric.__class__.__name__: 0}
                for key in results:
                    name = f'{task_name}_{key}'
                    if name in results:
                        """Inspired from https://github.com/open-
                        mmlab/mmengine/ bl ob/ed20a9cba52ceb371f7c825131636b9e2
                        747172e/mmengine/evalua tor/evaluator.py#L84-L87."""
                        raise ValueError(
                            'There are multiple metric results with the same'
                            f'metric name {name}. Please make sure all metrics'
                            'have different prefixes.')
                    metrics[name] = results[key]
        return metrics
