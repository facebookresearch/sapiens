# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.reduction import ForkingPickler
from typing import Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from .utils import LABEL_TYPE, SCORE_TYPE, format_label, format_score


class DataSample(BaseDataElement):
    """A general data structure interface.

    It's used as the interface between different components.

    The following fields are convention names in MMPretrain, and we will set or
    get these fields in data transforms, models, and metrics if needed. You can
    also set any new fields for your need.

    Meta fields:
        img_shape (Tuple): The shape of the corresponding input image.
        ori_shape (Tuple): The original shape of the corresponding image.
        sample_idx (int): The index of the sample in the dataset.
        num_classes (int): The number of all categories.

    Data fields:
        gt_label (tensor): The ground truth label.
        gt_score (tensor): The ground truth score.
        pred_label (tensor): The predicted label.
        pred_score (tensor): The predicted score.
        mask (tensor): The mask used in masked image modeling.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import DataSample
        >>>
        >>> img_meta = dict(img_shape=(960, 720), num_classes=5)
        >>> data_sample = DataSample(metainfo=img_meta)
        >>> data_sample.set_gt_label(3)
        >>> print(data_sample)
        <DataSample(
        META INFORMATION
            num_classes: 5
            img_shape: (960, 720)
        DATA FIELDS
            gt_label: tensor([3])
        ) at 0x7ff64c1c1d30>
        >>>
        >>> # For multi-label data
        >>> data_sample = DataSample().set_gt_label([0, 1, 4])
        >>> print(data_sample)
        <DataSample(
        DATA FIELDS
            gt_label: tensor([0, 1, 4])
        ) at 0x7ff5b490e100>
        >>>
        >>> # Set one-hot format score
        >>> data_sample = DataSample().set_pred_score([0.1, 0.1, 0.6, 0.1])
        >>> print(data_sample)
        <DataSample(
        META INFORMATION
            num_classes: 4
        DATA FIELDS
            pred_score: tensor([0.1000, 0.1000, 0.6000, 0.1000])
        ) at 0x7ff5b48ef6a0>
        >>>
        >>> # Set custom field
        >>> data_sample = DataSample()
        >>> data_sample.my_field = [1, 2, 3]
        >>> print(data_sample)
        <DataSample(
        DATA FIELDS
            my_field: [1, 2, 3]
        ) at 0x7f8e9603d3a0>
        >>> print(data_sample.my_field)
        [1, 2, 3]
    """

    def set_gt_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_gt_score(self, value: SCORE_TYPE) -> 'DataSample':
        """Set ``gt_score``."""
        score = format_score(value)
        self.set_field(score, 'gt_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'DataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE):
        """Set ``pred_label``."""
        score = format_score(value)
        self.set_field(score, 'pred_score', dtype=torch.Tensor)
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    def set_mask(self, value: Union[torch.Tensor, np.ndarray]):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            raise TypeError(f'Invalid mask type {type(value)}')
        self.set_field(value, 'mask', dtype=torch.Tensor)
        return self

    def __repr__(self) -> str:
        """Represent the object."""

        def dump_items(items, prefix=''):
            return '\n'.join(f'{prefix}{k}: {v}' for k, v in items)

        repr_ = ''
        if len(self._metainfo_fields) > 0:
            repr_ += '\n\nMETA INFORMATION\n'
            repr_ += dump_items(self.metainfo_items(), prefix=' ' * 4)
        if len(self._data_fields) > 0:
            repr_ += '\n\nDATA FIELDS\n'
            repr_ += dump_items(self.items(), prefix=' ' * 4)

        repr_ = f'<{self.__class__.__name__}({repr_}\n\n) at {hex(id(self))}>'
        return repr_


def _reduce_datasample(data_sample):
    """reduce DataSample."""
    attr_dict = data_sample.__dict__
    convert_keys = []
    for k, v in attr_dict.items():
        if isinstance(v, torch.Tensor):
            attr_dict[k] = v.numpy()
            convert_keys.append(k)
    return _rebuild_datasample, (attr_dict, convert_keys)


def _rebuild_datasample(attr_dict, convert_keys):
    """rebuild DataSample."""
    data_sample = DataSample()
    for k in convert_keys:
        attr_dict[k] = torch.from_numpy(attr_dict[k])
    data_sample.__dict__ = attr_dict
    return data_sample


# Due to the multi-processing strategy of PyTorch, DataSample may consume many
# file descriptors because it contains multiple tensors. Here we overwrite the
# reduce function of DataSample in ForkingPickler and convert these tensors to
# np.ndarray during pickling. It may slightly influence the performance of
# dataloader.
ForkingPickler.register(DataSample, _reduce_datasample)
