# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import os
import random
import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmcv.transforms import to_tensor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

Number = Union[int, float]

@TRANSFORMS.register_module()
class PackStereoCorrespondencesInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction',
                            'K', 'M')):
        self.meta_keys = meta_keys

    def transform(self, results: dict, min_depth=1e-2, max_depth=5) -> dict:
        packed_results = dict()
        results1 = results['results1']
        results2 = results['results2']

        ## overlap_percentage is too small. resample again.
        if len(results1['pixel_coords1']) == 0:
            return None

        if 'img' in results1:
            img = results1['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs1'] = img

        if 'img' in results2:
            img = results2['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs2'] = img

        data_sample1 = SegDataSample()
        data_sample2 = SegDataSample()

        img_meta1 = {}
        img_meta2 = {}

        for key in self.meta_keys:
            if key in results1:
                img_meta1[key] = results1[key]

            if key in results2:
                img_meta2[key] = results2[key]

        data_sample1.set_metainfo(img_meta1)
        data_sample2.set_metainfo(img_meta2)

        packed_results['data_samples1'] = data_sample1
        packed_results['data_samples2'] = data_sample2

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
