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
import numpy as np
import os
import cv2
from .transforms.pointmap_transforms import PointmapRandomFlip

##----------------------------------------------------------------------
@DATASETS.register_module()
class StereoPointmapCombinedDataset(BaseSegDataset):
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

        super(StereoPointmapCombinedDataset, self).__init__(pipeline=pipeline, **kwargs)
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        assert len(self.datasets) > 0

        return

    @property
    def metainfo(self):
        return deepcopy(self._metainfo)

    @force_full_init
    def __len__(self):
        return self._len

    def _get_subset_index(self, index: int) -> Tuple[int, int]:
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
        data_info, other_data_info = self.get_data_info(idx)

        # Apply the first N-3 transforms to data_info and other_data_info
        for transform in self.pipeline.transforms[:-3]:
            data_info = transform(data_info)

            if isinstance(transform, PointmapRandomFlip):
                if data_info['flip'] == True:
                    other_data_info['flip'] = True
                    other_data_info['flip_direction'] = data_info['flip_direction']
                    other_data_info = transform._flip(other_data_info) ## force the flip to the other_data_info
                else:
                    other_data_info['flip'] = False
                    other_data_info['flip_direction'] = None
            else:
                other_data_info = transform(other_data_info)

        if (data_info['mask'] > 0).sum() <= 1e4 or (other_data_info['mask'] > 0).sum() <= 1e4: ## atleast greater than 100 x 100 pixels
            return None

        data_info = {'results1': data_info, 'results2': other_data_info} ## pack the data_info and other_data_info

        for transform in self.pipeline.transforms[-3:]:
            data_info = transform(data_info)

        return data_info

    def get_data_info(self, idx: int) -> dict:
        subset_idx, sample_idx = self._get_subset_index(idx)
        data_info = self.datasets[subset_idx][sample_idx]
        return data_info

    def full_init(self):
        if self._fully_initialized:
            return

        for dataset in self.datasets:
            dataset.full_init()
        self._fully_initialized = True
