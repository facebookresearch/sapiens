# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.utils import is_seq_of

from mmpretrain.registry import METRICS
from mmpretrain.structures import label_to_onehot
from .single_label import to_tensor


@METRICS.register_module()
class RetrievalRecall(BaseMetric):
    r"""Recall evaluation metric for image retrieval.

    Args:
        topk (int | Sequence[int]): If the ground truth label matches one of
            the best **k** predictions, the sample will be regard as a positive
            prediction. If the parameter is a tuple, all of top-k recall will
            be calculated and outputted together. Defaults to 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        Use in the code:

        >>> import torch
        >>> from mmpretrain.evaluation import RetrievalRecall
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [[0], [1], [2], [3]]
        >>> y_true = [[0, 1], [2], [1], [0, 3]]
        >>> RetrievalRecall.calculate(
        >>>     y_pred, y_true, topk=1, pred_indices=True, target_indices=True)
        [tensor([50.])]
        >>> # Calculate the recall@1 and recall@5 for non-indices input.
        >>> y_score = torch.rand((1000, 10))
        >>> import torch.nn.functional as F
        >>> y_true = F.one_hot(torch.arange(0, 1000) % 10, num_classes=10)
        >>> RetrievalRecall.calculate(y_score, y_true, topk=(1, 5))
        [tensor(9.3000), tensor(48.4000)]
        >>>
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmpretrain.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label([0, 1]).set_pred_score(
        ...     torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=RetrievalRecall(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {'retrieval/Recall@1': 20.700000762939453,
         'retrieval/Recall@5': 78.5999984741211}

        Use in OpenMMLab configs:

        .. code:: python

            val_evaluator = dict(type='RetrievalRecall', topk=(1, 5))
            test_evaluator = val_evaluator
    """
    default_prefix: Optional[str] = 'retrieval'

    def __init__(self,
                 topk: Union[int, Sequence[int]],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        topk = (topk, ) if isinstance(topk, int) else topk

        for k in topk:
            if k <= 0:
                raise ValueError('`topk` must be a ingter larger than 0 '
                                 'or seq of ingter larger than 0.')

        self.topk = topk
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]):
        """Process one batch of data and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_score = data_sample['pred_score'].clone()
            gt_label = data_sample['gt_label']

            if 'gt_score' in data_sample:
                target = data_sample.get('gt_score').clone()
            else:
                num_classes = pred_score.size()[-1]
                target = label_to_onehot(gt_label, num_classes)

            # Because the retrieval output logit vector will be much larger
            # compared to the normal classification, to save resources, the
            # evaluation results are computed each batch here and then reduce
            #  all results at the end.
            result = RetrievalRecall.calculate(
                pred_score.unsqueeze(0), target.unsqueeze(0), topk=self.topk)
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        result_metrics = dict()
        for i, k in enumerate(self.topk):
            recall_at_k = sum([r[i].item() for r in results]) / len(results)
            result_metrics[f'Recall@{k}'] = recall_at_k

        return result_metrics

    @staticmethod
    def calculate(pred: Union[np.ndarray, torch.Tensor],
                  target: Union[np.ndarray, torch.Tensor],
                  topk: Union[int, Sequence[int]],
                  pred_indices: (bool) = False,
                  target_indices: (bool) = False) -> float:
        """Calculate the average recall.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            topk (int, Sequence[int]): Predictions with the k-th highest
                scores are considered as positive.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. Defaults to False.

        Returns:
            List[float]: the average recalls.
        """
        topk = (topk, ) if isinstance(topk, int) else topk
        for k in topk:
            if k <= 0:
                raise ValueError('`topk` must be a ingter larger than 0 '
                                 'or seq of ingter larger than 0.')

        max_keep = max(topk)
        pred = _format_pred(pred, max_keep, pred_indices)
        target = _format_target(target, target_indices)

        assert len(pred) == len(target), (
            f'Length of `pred`({len(pred)}) and `target` ({len(target)}) '
            f'must be the same.')

        num_samples = len(pred)
        results = []
        for k in topk:
            recalls = torch.zeros(num_samples)
            for i, (sample_pred,
                    sample_target) in enumerate(zip(pred, target)):
                sample_pred = np.array(to_tensor(sample_pred).cpu())
                sample_target = np.array(to_tensor(sample_target).cpu())
                recalls[i] = int(np.in1d(sample_pred[:k], sample_target).max())
            results.append(recalls.mean() * 100)
        return results


@METRICS.register_module()
class RetrievalAveragePrecision(BaseMetric):
    r"""Calculate the average precision for image retrieval.

    Args:
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive.
        mode (str, optional): The mode to calculate AP, choose from
                'IR'(information retrieval) and 'integrate'. Defaults to 'IR'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Note:
        If the ``mode`` set to 'IR', use the stanford AP calculation of
        information retrieval as in wikipedia page[1]; if set to 'integrate',
        the method implemented integrates over the precision-recall curve
        by averaging two adjacent precision points, then multiplying by the
        recall step like mAP in Detection task. This is the convention for
        the Revisited Oxford/Paris datasets[2].

    References:
        [1] `Wikipedia entry for the Average precision <https://en.wikipedia.
        org/wiki/Evaluation_measures_(information_retrieval)#Average_precision>`_

        [2] `The Oxford Buildings Dataset
        <https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/>`_

    Examples:
        Use in code:

        >>> import torch
        >>> import numpy as np
        >>> from mmcls.evaluation import RetrievalAveragePrecision
        >>> # using index format inputs
        >>> pred = [ torch.Tensor([idx for idx in range(100)]) ] * 3
        >>> target = [[0, 3, 6, 8, 35], [1, 2, 54, 105], [2, 42, 205]]
        >>> RetrievalAveragePrecision.calculate(pred, target, 10, True, True)
        29.246031746031747
        >>> # using tensor format inputs
        >>> pred = np.array([np.linspace(0.95, 0.05, 10)] * 2)
        >>> target = torch.Tensor([[1, 0, 1, 0, 0, 1, 0, 0, 1, 1]] * 2)
        >>> RetrievalAveragePrecision.calculate(pred, target, 10)
        62.222222222222214

        Use in OpenMMLab config files:

        .. code:: python

            val_evaluator = dict(type='RetrievalAveragePrecision', topk=100)
            test_evaluator = val_evaluator
    """

    default_prefix: Optional[str] = 'retrieval'

    def __init__(self,
                 topk: Optional[int] = None,
                 mode: Optional[str] = 'IR',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        if topk is None or (isinstance(topk, int) and topk <= 0):
            raise ValueError('`topk` must be a ingter larger than 0.')

        mode_options = ['IR', 'integrate']
        assert mode in mode_options, \
            f'Invalid `mode` argument, please specify from {mode_options}.'

        self.topk = topk
        self.mode = mode
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]):
        """Process one batch of data and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_score = data_sample.get('pred_score').clone()

            if 'gt_score' in data_sample:
                target = data_sample.get('gt_score').clone()
            else:
                gt_label = data_sample.get('gt_label')
                num_classes = pred_score.size()[-1]
                target = label_to_onehot(gt_label, num_classes)

            # Because the retrieval output logit vector will be much larger
            # compared to the normal classification, to save resources, the
            # evaluation results are computed each batch here and then reduce
            #  all results at the end.
            result = RetrievalAveragePrecision.calculate(
                pred_score.unsqueeze(0),
                target.unsqueeze(0),
                self.topk,
                mode=self.mode)
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        result_metrics = dict()
        result_metrics[f'mAP@{self.topk}'] = np.mean(self.results).item()

        return result_metrics

    @staticmethod
    def calculate(pred: Union[np.ndarray, torch.Tensor],
                  target: Union[np.ndarray, torch.Tensor],
                  topk: Optional[int] = None,
                  pred_indices: (bool) = False,
                  target_indices: (bool) = False,
                  mode: str = 'IR') -> float:
        """Calculate the average precision.
        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, M)`` or a sequence of index/onehot
                format labels.
            topk (int, optional): Predictions with the k-th highest scores
                 are considered as positive.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. Defaults to False.
            mode (Optional[str]): The mode to calculate AP, choose from
                'IR'(information retrieval) and 'integrate'. Defaults to 'IR'.

        Note:
            If the ``mode`` set to 'IR', use the stanford AP calculation of
            information retrieval as in wikipedia page; if set to 'integrate',
            the method implemented integrates over the precision-recall curve
            by averaging two adjacent precision points, then multiplying by the
            recall step like mAP in Detection task. This is the convention for
            the Revisited Oxford/Paris datasets.

        Returns:
            float: the average precision of the query image.

        References:
            [1] `Wikipedia entry for Average precision(information_retrieval)
            <https://en.wikipedia.org/wiki/Evaluation_measures_

            (information_retrieval)#Average_precision>`_
            [2] `The Oxford Buildings Dataset <https://www.robots.ox.ac.uk/
            ~vgg/data/oxbuildings/`_
        """
        if topk is None or (isinstance(topk, int) and topk <= 0):
            raise ValueError('`topk` must be a ingter larger than 0.')

        mode_options = ['IR', 'integrate']
        assert mode in mode_options, \
            f'Invalid `mode` argument, please specify from {mode_options}.'

        pred = _format_pred(pred, topk, pred_indices)
        target = _format_target(target, target_indices)

        assert len(pred) == len(target), (
            f'Length of `pred`({len(pred)}) and `target` ({len(target)}) '
            f'must be the same.')

        num_samples = len(pred)
        aps = np.zeros(num_samples)
        for i, (sample_pred, sample_target) in enumerate(zip(pred, target)):
            aps[i] = _calculateAp_for_sample(sample_pred, sample_target, mode)

        return aps.mean()


def _calculateAp_for_sample(pred, target, mode):
    pred = np.array(to_tensor(pred).cpu())
    target = np.array(to_tensor(target).cpu())

    num_preds = len(pred)

    # TODO: use ``torch.isin`` in torch1.10.
    positive_ranks = np.arange(num_preds)[np.in1d(pred, target)]

    ap = 0
    for i, rank in enumerate(positive_ranks):
        if mode == 'IR':
            precision = (i + 1) / (rank + 1)
            ap += precision
        elif mode == 'integrate':
            # code are modified from https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp # noqa:
            old_precision = i / rank if rank > 0 else 1
            cur_precision = (i + 1) / (rank + 1)
            prediction = (old_precision + cur_precision) / 2
            ap += prediction
    ap = ap / len(target)

    return ap * 100


def _format_pred(label, topk=None, is_indices=False):
    """format various label to List[indices]."""
    if is_indices:
        assert isinstance(label, Sequence),  \
                '`pred` must be Sequence of indices when' \
                f' `pred_indices` set to True, but get {type(label)}'
        for i, sample_pred in enumerate(label):
            assert is_seq_of(sample_pred, int) or isinstance(
                sample_pred, (np.ndarray, torch.Tensor)), \
                '`pred` should be Sequence of indices when `pred_indices`' \
                f'set to True. but pred[{i}] is {sample_pred}'
            if topk:
                label[i] = sample_pred[:min(topk, len(sample_pred))]
        return label
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif not isinstance(label, torch.Tensor):
        raise TypeError(f'The pred must be type of torch.tensor, '
                        f'np.ndarray or Sequence but get {type(label)}.')
    topk = topk if topk else label.size()[-1]
    _, indices = label.topk(topk)
    return indices


def _format_target(label, is_indices=False):
    """format various label to List[indices]."""
    if is_indices:
        assert isinstance(label, Sequence),  \
                '`target` must be Sequence of indices when' \
                f' `target_indices` set to True, but get {type(label)}'
        for i, sample_gt in enumerate(label):
            assert is_seq_of(sample_gt, int) or isinstance(
                sample_gt, (np.ndarray, torch.Tensor)), \
                '`target` should be Sequence of indices when ' \
                f'`target_indices` set to True. but target[{i}] is {sample_gt}'
        return label

    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif isinstance(label, Sequence) and not mmengine.is_str(label):
        label = torch.tensor(label)
    elif not isinstance(label, torch.Tensor):
        raise TypeError(f'The pred must be type of torch.tensor, '
                        f'np.ndarray or Sequence but get {type(label)}.')

    indices = [sample_gt.nonzero().squeeze(-1) for sample_gt in label]
    return indices
