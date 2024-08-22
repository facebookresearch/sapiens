# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.utils import is_str

if hasattr(torch, 'tensor_split'):
    tensor_split = torch.tensor_split
else:
    # A simple implementation of `tensor_split`.
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs


LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The foramtted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence): Score values.

    Returns:
        :obj:`torch.Tensor`: The foramtted score tensor.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def cat_batch_labels(elements: List[torch.Tensor]):
    """Concat a batch of label tensor to one tensor.

    Args:
        elements (List[tensor]): A batch of labels.

    Returns:
        Tuple[torch.Tensor, List[int]]: The first item is the concated label
        tensor, and the second item is the split indices of every sample.
    """
    labels = []
    splits = [0]
    for element in elements:
        labels.append(element)
        splits.append(splits[-1] + element.size(0))
    batch_label = torch.cat(labels)
    return batch_label, splits[1:-1]


def batch_label_to_onehot(batch_label, split_indices, num_classes):
    """Convert a concated label tensor to onehot format.

    Args:
        batch_label (torch.Tensor): A concated label tensor from multiple
            samples.
        split_indices (List[int]): The split indices of every sample.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import batch_label_to_onehot
        >>> # Assume a concated label from 3 samples.
        >>> # label 1: [0, 1], label 2: [0, 2, 4], label 3: [3, 1]
        >>> batch_label = torch.tensor([0, 1, 0, 2, 4, 3, 1])
        >>> split_indices = [2, 5]
        >>> batch_label_to_onehot(batch_label, split_indices, num_classes=5)
        tensor([[1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]])
    """
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [
        sparse_onehot.sum(0)
        for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)
    ]
    return torch.stack(onehot_list)


def label_to_onehot(label: LABEL_TYPE, num_classes: int):
    """Convert a label to onehot format tensor.

    Args:
        label (LABEL_TYPE): Label value.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import label_to_onehot
        >>> # Single-label
        >>> label_to_onehot(1, num_classes=5)
        tensor([0, 1, 0, 0, 0])
        >>> # Multi-label
        >>> label_to_onehot([0, 2, 3], num_classes=5)
        tensor([1, 0, 1, 1, 0])
    """
    label = format_label(label)
    sparse_onehot = F.one_hot(label, num_classes)
    return sparse_onehot.sum(0)
