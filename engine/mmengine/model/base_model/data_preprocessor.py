# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmengine.structures import BaseDataElement
from mmengine.utils import is_seq_of
from ..utils import stack_batch

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


@MODELS.register_module()
class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for copying data to the target device.

    Subclasses inherit from ``BaseDataPreprocessor`` could override the
    forward method to implement custom data pre-processing, such as
    batch-resize, MixUp, or CutMix.

    Args:
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version 0.3.0.

    Note:
        Data dictionary returned by dataloader must be a dict and at least
        contain the ``inputs`` key.
    """

    def __init__(self, non_blocking: Optional[bool] = False):
        super().__init__()
        self._non_blocking = non_blocking
        self._device = torch.device('cpu')

    def cast_data(self, data: CastData) -> CastData:
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data.to(self.device, non_blocking=self._non_blocking)
        else:
            return data

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Preprocesses the data into the model input format.

        After the data pre-processing of :meth:`cast_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        return self.cast_data(data)  # type: ignore

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and 'npu' in args[0]:
            args = tuple(
                [list(args)[0].replace('npu', torch.npu.native_device)])
        if kwargs and 'npu' in str(kwargs.get('device', '')):
            kwargs['device'] = kwargs['device'].replace(
                'npu', torch.npu.native_device)

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._device = torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.cuda.current_device())
        return super().cuda()

    def npu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.npu.current_device())
        return super().npu()

    def mlu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device(torch.mlu.current_device())
        return super().mlu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """
        self._device = torch.device('cpu')
        return super().cpu()


@MODELS.register_module()
class ImgDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for normalization and bgr to rgb conversion.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the model input. ``ImgDataPreprocessor`` provides the
    basic data pre-processing as follows

    - Collates and moves data to the target device.
    - Converts inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalizes image with defined std and mean.
    - Pads inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.

    For ``ImgDataPreprocessor``, the dimension of the single inputs must be
    (3, H, W).

    Note:
        ``ImgDataPreprocessor`` and its subclass is built in the
        constructor of :class:`BaseDataset`.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``bgr_to_rgb=True`` it means the mean value of R,
            G, B channels. If the length of `mean` is 1, it means all
            channels have the same mean value, or the input is a gray image.
            If it is not specified, images will not be normalized. Defaults
            None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``bgr_to_rgb=True`` it means the standard
            deviation of R, G, B channels. If the length of `std` is 1,
            it means all channels have the same standard deviation, or the
            input is a gray image.  If it is not specified, images will
            not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device.
            New in version v0.3.0.

    Note:
        if images do not need to be normalized, `std` and `mean` should be
        both set to None, otherwise both of them should be set to a tuple of
        corresponding values.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean)} values')
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)} values')  # type: ignore
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data
