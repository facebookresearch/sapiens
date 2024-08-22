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

try:
    from libcom import color_transfer
    LIBCOM_INSTALLED = True
except ImportError:
    color_transfer = None
    LIBCOM_INSTALLED = False

try:
    import albumentations
    from albumentations import Compose
    ALBU_INSTALLED = True
except ImportError:
    albumentations = None
    Compose = None
    ALBU_INSTALLED = False


@TRANSFORMS.register_module()
class HDRIResize(BaseTransform):
    def __init__(self, scale: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.scale = scale


    def transform(self, results: Dict) -> Dict:

        target_width = self.scale[0]
        target_height = self.scale[1]

        results['img'] = cv2.resize(
                                    results['img'],
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_LINEAR  # Good for image upsampling/downsampling
                                )

        return results

@TRANSFORMS.register_module()
class GenerateHDRITarget(BaseTransform):
    def __init__(self):
        return

    def transform(self, results: dict) -> dict:
        gt_hdri = results['gt_hdri'] ## lighting intensity
        results['gt_depth_map'] = gt_hdri
        return results

    def __repr__(self):
        return self.__class__.__name__

@TRANSFORMS.register_module()
class PackHDRIInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction',)):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()

        ## is actually the 16 x 32 x 3 HDR image
        if 'gt_depth_map' in results:
            gt_depth_map = results['gt_depth_map']
            gt_depth_map = gt_depth_map.transpose(2, 0, 1) ## 3 x H x W, 3 x 16 x 32

            gt_depth_data = dict(data=to_tensor(gt_depth_map.copy()))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
