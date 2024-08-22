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
class PointmapRandomFlip(MMCV_RandomFlip):
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        assert results['flip_direction'] == 'horizontal'

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip seg map
        results['mask'] = mmcv.imflip(results['mask'], direction=results['flip_direction'])

        if 'gt_depth' in results.keys():
            results['gt_depth'] = mmcv.imflip(results['gt_depth'], direction=results['flip_direction'])

        if 'K' in results.keys():
            # Flip the principal point for the left-right flipped image
            results['K'][0, 2] = img_shape[1] - results['K'][0, 2] - 1

        if 'M' in results.keys():
            # Flip the sign of the first column of the extrinsics matrix
            results['M'][0, :] = -results['M'][0, :]

        return results

@TRANSFORMS.register_module()
class RandomPointmapResizeCompensate(BaseTransform):

    def __init__(
        self):
        super().__init__()


    def transform(self, results: dict) -> dict:
        assert 'scale' in results.keys() and 'scale_factor' in results.keys()
        keep_ratio = results['keep_ratio']

        target_height, target_width = results['img'].shape[:2]
        original_height, original_width = results['mask'].shape[:2]

        # Resize mask to the target size
        mask = results['mask']

        if keep_ratio:
            mask_resized, scale_factor = mmcv.imrescale(mask, results['scale'], interpolation='nearest', return_scale=True, backend='cv2')
        else:
            mask_resized, w_scale, h_scale = mmcv.imresize(mask, results['scale'], interpolation='nearest', return_scale=True, backend='cv2')

        results['mask'] = mask_resized
        assert results['mask'].shape[0] == results['img'].shape[0] and results['mask'].shape[1] == results['img'].shape[1]

        if 'gt_depth' in results:
            gt_depth = results['gt_depth']

            ## keep_ratio is true from RandomResize in the config
            if keep_ratio:
                gt_depth_resized, scale_factor = mmcv.imrescale(
                    gt_depth,
                    results['scale'],
                    interpolation='nearest',
                    return_scale=True,
                    backend='cv2')
            else:
                gt_depth_resized, w_scale, h_scale = mmcv.imresize(
                    gt_depth,
                    results['scale'],
                    interpolation='nearest',
                    return_scale=True,
                    backend='cv2')

            results['gt_depth'] = gt_depth_resized
            assert results['gt_depth'].shape[0] == results['img'].shape[0] and results['gt_depth'].shape[1] == results['img'].shape[1]

        if 'K' in results:
            K_new = results['K'].copy()

            # Adjust the intrinsic matrix for the new image dimensions
            K_new[0, 0] = K_new[0, 0] * target_width / original_width
            K_new[0, 2] = K_new[0, 2] * target_width / original_width
            K_new[1, 1] = K_new[1, 1] * target_height / original_height
            K_new[1, 2] = K_new[1, 2] * target_height / original_height

            results['K'] = K_new

        return results

    def __repr__(self):
        return self.__class__.__name__

@TRANSFORMS.register_module()
class RandomPointmapCrop(BaseTransform):

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]]):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']

        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop depth and crop mask and crop normal
        for key in ['gt_depth', 'mask']:

            if key not in results.keys():
                continue

            results[key] = self.crop(results[key], crop_bbox)

        if 'K' in results:
            K_new = results['K'].copy()
            # Adjust the principal point according to the crop
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
            K_new[0, 2] = K_new[0, 2] - crop_x1
            K_new[1, 2] = K_new[1, 2] - crop_y1
            results['K'] = K_new

        results['img'] = img
        results['img_shape'] = img.shape[:2]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@TRANSFORMS.register_module()
class PointmapResize(BaseTransform):
    def __init__(self, scale: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.scale = scale


    def transform(self, results: Dict) -> Dict:
        target_width = self.scale[0]
        target_height = self.scale[1]

        original_width = results['img'].shape[1]
        original_height = results['img'].shape[0]

        ## ---------------------------------
        results['img'] = cv2.resize(
                                    results['img'],
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_LINEAR  # Good for image upsampling/downsampling
                                )

        results['mask'] = cv2.resize(
                                    results['mask'],
                                    (target_width, target_height),
                                    interpolation=cv2.INTER_NEAREST
                                )


        if 'K' in results:
            K_new = results['K'].copy()

            # Adjust the intrinsic matrix for the new image dimensions
            K_new[0, 0] = K_new[0, 0] * target_width / original_width ## fx
            K_new[0, 2] = K_new[0, 2] * target_width / original_width ## cx
            K_new[1, 1] = K_new[1, 1] * target_height / original_height ## fy
            K_new[1, 2] = K_new[1, 2] * target_height / original_height ## cy


            results['K'] = K_new

        if 'gt_depth' in results:
            results['gt_depth'] = cv2.resize(
                                        results['gt_depth'],
                                        (target_width, target_height),
                                        interpolation=cv2.INTER_NEAREST
                                    )

        return results


@TRANSFORMS.register_module()
class GeneratePointmapTarget(BaseTransform):
    def __init__(self, background_val=-1000):
        self.background_val = background_val
        return

    def transform(self, results: dict) -> dict:
        mask = results['mask']
        depth = results['gt_depth']
        K = results['K']

        # Create a grid of pixel coordinates
        height, width = depth.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))

        z = depth
        x = (cols - K[0, 2]) * z / K[0, 0]
        y = (rows - K[1, 2]) * z / K[1, 1]
        gt_pointmap = np.stack([x, y, z], axis=-1) ## H x W x 3

        gt_pointmap[mask == 0] = self.background_val ## set the background to -1000. the loss uses a threshold of -10 to pick up.
        results['gt_depth_map'] = gt_pointmap ## the key gt_depth_map is used for legacy reasons.

        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class PackPointmapInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction',
                            'K', 'M')):
        self.meta_keys = meta_keys

    def transform(self, results: dict, min_depth=1e-2, max_depth=5) -> dict:
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

        ## stereo pointmap
        if 'gt_depth_map' in results:
            mask = results['mask']
            gt_depth_map = results['gt_depth_map'] ## 1024 x 768 x 3
            gt_depth_map[mask > 0][2] = np.clip(gt_depth_map[mask > 0][2], min_depth, max_depth)
            gt_depth_map = gt_depth_map.transpose(2, 0, 1) ## 3 x H x W, 3 x 1024 x 1024

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

@TRANSFORMS.register_module()
class PadPointmap(BaseTransform):
    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_to_square: bool = False,
                 pad_val: Union[Number, dict] = dict(img=0, seg=255),
                 padding_mode: str = 'constant') -> None:
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, int):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(pad_val, dict), 'pad_val '
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)

        size = None
        if self.pad_to_square:
            max_size = max(results['img'].shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'].shape[0], results['img'].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results['img'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['img'].shape[2]))
        padded_img = mmcv.impad(
            results['img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        original_height = results['img'].shape[0]
        original_width = results['img'].shape[1]

        width = max(padded_img.shape[1] - original_width, 0)
        height = max(padded_img.shape[0] - original_height, 0)

        padding_left = 0
        padding_right = width
        padding_top = 0
        padding_bottom = height

        padding_size = (padding_left, padding_right, padding_top, padding_bottom)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['img_shape'] = padded_img.shape[:2]
        results['padding_size'] = padding_size

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get('gt_seg_map', None) is not None:
            pad_val = self.pad_val.get('seg', 255)
            if isinstance(pad_val, int) and results['gt_seg_map'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['gt_seg_map'].shape[2]))
            results['gt_seg_map'] = mmcv.impad(
                results['gt_seg_map'],
                shape=results['pad_shape'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'padding_mode={self.padding_mode})'
        return repr_str
