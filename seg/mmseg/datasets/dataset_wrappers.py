# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
from copy import deepcopy
from typing import List, Optional, Sequence, Union, Any, Tuple, Callable

from mmengine.dataset import ConcatDataset, force_full_init
from mmseg.registry import DATASETS, TRANSFORMS

from mmseg.datasets import BaseSegDataset
from mmengine.registry import build_from_cfg

##----------------------------------------------------------------------
@DATASETS.register_module()
class CombinedDataset(BaseSegDataset):
    """A wrapper of combined dataset.
    Args:
        metainfo (dict): The meta information of combined dataset.
        datasets (list): The configs of datasets to be combined.
        pipeline (list, optional): Processing pipeline. Defaults to [].
    """

    def __init__(self,
                 metainfo: dict,
                 datasets: list,
                 pipeline: List[Union[dict, Callable]] = [],
                 **kwargs):

        self.datasets = []

        for cfg in datasets:
            dataset = build_from_cfg(cfg, DATASETS)
            self.datasets.append(dataset)

        self._lens = [len(dataset) for dataset in self.datasets]
        self._len = sum(self._lens)

        super(CombinedDataset, self).__init__(pipeline=pipeline, **kwargs)
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        assert len(self.datasets) > 0

        self._metainfo['classes'] = self.datasets[0]._metainfo['classes']
        self._metainfo['palette'] = self.datasets[0]._metainfo['palette']

        return

    @property
    def metainfo(self):
        return deepcopy(self._metainfo)

    @force_full_init
    def __len__(self):
        return self._len

    def _get_subset_index(self, index: int) -> Tuple[int, int]:
        """Given a data sample's global index, return the index of the sub-
        dataset the data sample belongs to, and the local index within that
        sub-dataset.
        Args:
            index (int): The global data sample index
        Returns:
            tuple[int, int]:
            - subset_index (int): The index of the sub-dataset
            - local_index (int): The index of the data sample within
                the sub-dataset
        """
        if index >= len(self) or index < -len(self):
            raise ValueError(
                f'index({index}) is out of bounds for dataset with '
                f'length({len(self)}).')

        if index < 0:
            index = index + len(self)

        subset_index = 0
        while index >= self._lens[subset_index]:
            index -= self._lens[subset_index]
            subset_index += 1
        return subset_index, index

    def prepare_data(self, idx: int) -> Any:
        """Get data processed by ``self.pipeline``.The source dataset is
        depending on the index.
        Args:
            idx (int): The index of ``data_info``.
        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        ## check if sample belongs to the goliath dataset
        subset_idx, sample_idx = self._get_subset_index(idx)

        # try:
        #     transformed_data_info = self.pipeline(data_info)
        # except Exception as e:
        #     print('Exception raised during processing sample {}, subset idx:{}'.format(idx, subset_idx), e)
        #     return None

        transformed_data_info = self.pipeline(data_info)

        return transformed_data_info

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.
        Args:
            idx (int): Global index of ``CombinedDataset``.
        Returns:
            dict: The idx-th annotation of the datasets.
        """
        subset_idx, sample_idx = self._get_subset_index(idx)
        # Get data sample processed by ``subset.pipeline``
        data_info = self.datasets[subset_idx][sample_idx]

        return data_info

    def full_init(self):
        """Fully initialize all sub datasets."""

        if self._fully_initialized:
            return

        for dataset in self.datasets:
            dataset.full_init()
        self._fully_initialized = True


##----------------------------------------------------------------------
@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Args:
        dataset (ConcatDataset or dict): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset: Union[ConcatDataset, dict],
                 pipeline: Sequence[dict],
                 skip_type_keys: Optional[List[str]] = None,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`ConcatDataset` instance, but got {type(dataset)}')

        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self._metainfo = self.dataset.metainfo
        self.num_samples = len(self.dataset)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indices'):
                indices = transform.get_indices(self.dataset)
                if not isinstance(indices, collections.abc.Sequence):
                    indices = [indices]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indices
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
