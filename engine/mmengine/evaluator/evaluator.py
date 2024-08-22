# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterator, List, Optional, Sequence, Union

from mmengine.dataset import pseudo_collate
from mmengine.registry import EVALUATOR, METRICS
from mmengine.structures import BaseDataElement
from .metric import BaseMetric


@EVALUATOR.register_module()
class Evaluator:
    """Wrapper class to compose multiple :class:`BaseMetric` instances.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self._dataset_meta: Optional[dict] = None
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, dict):
                self.metrics.append(METRICS.build(metric))
            else:
                self.metrics.append(metric)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            metric.dataset_meta = dataset_meta

    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.process(data_batch, _data_samples)

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics

    def offline_evaluate(self,
                         data_samples: Sequence,
                         data: Optional[Sequence] = None,
                         chunk_size: int = 1):
        """Offline evaluate the dumped predictions on the given data .

        Args:
            data_samples (Sequence): All predictions and ground truth of the
                model and the validation set.
            data (Sequence, optional): All data of the validation set.
            chunk_size (int): The number of data samples and predictions to be
                processed in a batch.
        """

        # support chunking iterable objects
        def get_chunks(seq: Iterator, chunk_size=1):
            stop = False
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(seq))
                    except StopIteration:
                        stop = True
                        break
                if chunk:
                    yield chunk

        if data is not None:
            assert len(data_samples) == len(data), (
                'data_samples and data should have the same length, but got '
                f'data_samples length: {len(data_samples)} '
                f'data length: {len(data)}')
            data = get_chunks(iter(data), chunk_size)

        size = 0
        for output_chunk in get_chunks(iter(data_samples), chunk_size):
            if data is not None:
                data_chunk = pseudo_collate(next(data))  # type: ignore
            else:
                data_chunk = None
            size += len(output_chunk)
            self.process(output_chunk, data_chunk)
        return self.evaluate(size)
