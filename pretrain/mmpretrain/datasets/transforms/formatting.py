# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str
from PIL import Image

from mmpretrain.registry import TRANSFORMS
from mmpretrain.structures import DataSample, MultiTaskDataSample


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data.

    **Required Keys:**

    - ``input_key``
    - ``*algorithm_keys``
    - ``*meta_keys``

    **Deleted Keys:**

    All other keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmpretrain.structures.DataSample`): The
      annotation info of the sample.

    Args:
        input_key (str): The key of element to feed into the model forwarding.
            Defaults to 'img'.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
        meta_keys (Sequence[str]): The keys of meta information to be saved in
            the data sample. Defaults to :attr:`PackInputs.DEFAULT_META_KEYS`.

    .. admonition:: Default algorithm keys

        Besides the specified ``algorithm_keys``, we will set some default keys
        into the output data sample and do some formatting. Therefore, you
        don't need to set these keys in the ``algorithm_keys``.

        - ``gt_label``: The ground-truth label. The value will be converted
          into a 1-D tensor.
        - ``gt_score``: The ground-truth score. The value will be converted
          into a 1-D tensor.
        - ``mask``: The mask for some self-supervise tasks. The value will
          be converted into a tensor.

    .. admonition:: Default meta keys

        - ``sample_idx``: The id of the image sample.
        - ``img_path``: The path to the image file.
        - ``ori_shape``: The original shape of the image as a tuple (H, W).
        - ``img_shape``: The shape of the image after the pipeline as a
          tuple (H, W).
        - ``scale_factor``: The scale factor between the resized image and
          the original image.
        - ``flip``: A boolean indicating if image flip transform was used.
        - ``flip_direction``: The flipping direction.
    """

    DEFAULT_META_KEYS = ('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')

    def __init__(self,
                 input_key='img',
                 algorithm_keys=(),
                 meta_keys=DEFAULT_META_KEYS):
        self.input_key = input_key
        self.algorithm_keys = algorithm_keys
        self.meta_keys = meta_keys

    @staticmethod
    def format_input(input_):
        if isinstance(input_, list):
            return [PackInputs.format_input(item) for item in input_]
        elif isinstance(input_, np.ndarray):
            if input_.ndim == 2:  # For grayscale image.
                input_ = np.expand_dims(input_, -1)
            if input_.ndim == 3 and not input_.flags.c_contiguous:
                input_ = np.ascontiguousarray(input_.transpose(2, 0, 1))
                input_ = to_tensor(input_)
            elif input_.ndim == 3:
                # convert to tensor first to accelerate, see
                # https://github.com/open-mmlab/mmdetection/pull/9533
                input_ = to_tensor(input_).permute(2, 0, 1).contiguous()
            else:
                # convert input with other shape to tensor without permute,
                # like video input (num_crops, C, T, H, W).
                input_ = to_tensor(input_)
        elif isinstance(input_, Image.Image):
            input_ = F.pil_to_tensor(input_)
        elif not isinstance(input_, torch.Tensor):
            raise TypeError(f'Unsupported input type {type(input_)}.')

        return input_

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = dict()
        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = self.format_input(input_)

        data_sample = DataSample()

        # Set default keys
        if 'gt_label' in results:
            data_sample.set_gt_label(results['gt_label'])
        if 'gt_score' in results:
            data_sample.set_gt_score(results['gt_score'])
        if 'mask' in results:
            data_sample.set_mask(results['mask'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        for key in self.meta_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.input_key}', "
        repr_str += f'algorithm_keys={self.algorithm_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackMultiTaskInputs(BaseTransform):
    """Convert all image labels of multi-task dataset to a dict of tensor.

    Args:
        multi_task_fields (Sequence[str]):
        input_key (str):
        task_handlers (dict):
    """

    def __init__(self,
                 multi_task_fields,
                 input_key='img',
                 task_handlers=dict()):
        self.multi_task_fields = multi_task_fields
        self.input_key = input_key
        self.task_handlers = defaultdict(PackInputs)
        for task_name, task_handler in task_handlers.items():
            self.task_handlers[task_name] = TRANSFORMS.build(task_handler)

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        result = {'img_path': 'a.png', 'gt_label': {'task1': 1, 'task3': 3},
            'img': array([[[  0,   0,   0])
        """
        packed_results = dict()
        results = results.copy()

        if self.input_key in results:
            input_ = results[self.input_key]
            packed_results['inputs'] = PackInputs.format_input(input_)

        task_results = defaultdict(dict)
        for field in self.multi_task_fields:
            if field in results:
                value = results.pop(field)
                for k, v in value.items():
                    task_results[k].update({field: v})

        data_sample = MultiTaskDataSample()
        for task_name, task_result in task_results.items():
            task_handler = self.task_handlers[task_name]
            task_pack_result = task_handler({**results, **task_result})
            data_sample.set_field(task_pack_result['data_samples'], task_name)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        repr = self.__class__.__name__
        task_handlers = ', '.join(
            f"'{name}': {handler.__class__.__name__}"
            for name, handler in self.task_handlers.items())
        repr += f'(multi_task_fields={self.multi_task_fields}, '
        repr += f"input_key='{self.input_key}', "
        repr += f'task_handlers={{{task_handlers}}})'
        return repr


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose numpy array.

    **Required Keys:**

    - ``*keys``

    **Modified Keys:**

    - ``*keys``

    Args:
        keys (List[str]): The fields to convert to tensor.
        order (List[int]): The output dimensions order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def transform(self, results):
        """Method to transpose array."""
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module(('NumpyToPIL', 'ToPIL'))
class NumpyToPIL(BaseTransform):
    """Convert the image from OpenCV format to :obj:`PIL.Image.Image`.

    **Required Keys:**

    - ``img``

    **Modified Keys:**

    - ``img``

    Args:
        to_rgb (bool): Whether to convert img to rgb. Defaults to True.
    """

    def __init__(self, to_rgb: bool = False) -> None:
        self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        """Method to convert images to :obj:`PIL.Image.Image`."""
        img = results['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if self.to_rgb else img

        results['img'] = Image.fromarray(img)
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(to_rgb={self.to_rgb})'


@TRANSFORMS.register_module(('PILToNumpy', 'ToNumpy'))
class PILToNumpy(BaseTransform):
    """Convert img to :obj:`numpy.ndarray`.

    **Required Keys:**

    - ``img``

    **Modified Keys:**

    - ``img``

    Args:
        to_bgr (bool): Whether to convert img to rgb. Defaults to True.
        dtype (str, optional): The dtype of the converted numpy array.
            Defaults to None.
    """

    def __init__(self, to_bgr: bool = False, dtype=None) -> None:
        self.to_bgr = to_bgr
        self.dtype = dtype

    def transform(self, results: dict) -> dict:
        """Method to convert img to :obj:`numpy.ndarray`."""
        img = np.array(results['img'], dtype=self.dtype)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.to_bgr else img

        results['img'] = img
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + \
            f'(to_bgr={self.to_bgr}, dtype={self.dtype})'


@TRANSFORMS.register_module()
class Collect(BaseTransform):
    """Collect and only reserve the specified fields.

    **Required Keys:**

    - ``*keys``

    **Deleted Keys:**

    All keys except those in the argument ``*keys``.

    Args:
        keys (Sequence[str]): The keys of the fields to be collected.
    """

    def __init__(self, keys):
        self.keys = keys

    def transform(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
