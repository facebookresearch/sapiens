# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import (BaseDataPreprocessor, ImgDataPreprocessor,
                            stack_batch)

from mmpretrain.registry import MODELS
from mmpretrain.structures import (DataSample, MultiTaskDataSample,
                                   batch_label_to_onehot, cat_batch_labels,
                                   tensor_split)
from .batch_augments import RandomBatchAugment


@MODELS.register_module()
class ClsDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for classification tasks.

    Comparing with the :class:`mmengine.model.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        to_onehot (bool): Whether to generate one-hot format gt-labels and set
            to data samples. Defaults to False.
        num_classes (int, optional): The number of classes. Defaults to None.
        batch_augments (dict, optional): The batch augmentations settings,
            including "augments" and "probs". For more details, see
            :class:`mmpretrain.models.RandomBatchAugment`.
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                'preprocessing, please specify both `mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if batch_augments:
            self.batch_augments = RandomBatchAugment(**batch_augments)
            if not self.to_onehot:
                from mmengine.logging import MMLogger
                MMLogger.get_current_instance().info(
                    'Because batch augmentations are enabled, the data '
                    'preprocessor automatically enables the `to_onehot` '
                    'option to generate one-hot format labels.')
                self.to_onehot = True
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs = self.cast_data(data['inputs'])

        if isinstance(inputs, torch.Tensor):
            # The branch if use `default_collate` as the collate_fn in the
            # dataloader.

            # ------ To RGB ------
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)

            # -- Normalization ---
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant',
                               self.pad_value)
        else:
            # The branch if use `pseudo_collate` as the collate_fn in the
            # dataloader.

            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if self.to_rgb and input_.size(0) == 3:
                    input_ = input_.flip(0)

                # -- Normalization ---
                input_ = input_.float()
                if self._enable_normalize:
                    input_ = (input_ - self.mean) / self.std

                processed_inputs.append(input_)
            # Combine padding and stack
            inputs = stack_batch(processed_inputs, self.pad_size_divisor,
                                 self.pad_value)

        data_samples = data.get('data_samples', None)
        sample_item = data_samples[0] if data_samples is not None else None

        if isinstance(sample_item, DataSample):
            batch_label = None
            batch_score = None

            if 'gt_label' in sample_item:
                gt_labels = [sample.gt_label for sample in data_samples]
                batch_label, label_indices = cat_batch_labels(gt_labels)
                batch_label = batch_label.to(self.device)
            if 'gt_score' in sample_item:
                gt_scores = [sample.gt_score for sample in data_samples]
                batch_score = torch.stack(gt_scores).to(self.device)
            elif self.to_onehot and 'gt_label' in sample_item:
                assert batch_label is not None, \
                    'Cannot generate onehot format labels because no labels.'
                num_classes = self.num_classes or sample_item.get(
                    'num_classes')
                assert num_classes is not None, \
                    'Cannot generate one-hot format labels because not set ' \
                    '`num_classes` in `data_preprocessor`.'
                batch_score = batch_label_to_onehot(
                    batch_label, label_indices, num_classes).to(self.device)

            # ----- Batch Augmentations ----
            if (training and self.batch_augments is not None
                    and batch_score is not None):
                inputs, batch_score = self.batch_augments(inputs, batch_score)

            # ----- scatter labels and scores to data samples ---
            if batch_label is not None:
                for sample, label in zip(
                        data_samples, tensor_split(batch_label,
                                                   label_indices)):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)
        elif isinstance(sample_item, MultiTaskDataSample):
            data_samples = self.cast_data(data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}


@MODELS.register_module()
class SelfSupDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for operations, like normalization and bgr to rgb.

    Compared with the :class:`mmengine.ImgDataPreprocessor`, this module
    supports ``inputs`` as torch.Tensor or a list of torch.Tensor.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)

        self._channel_conversion = to_rgb or bgr_to_rgb or rgb_to_bgr

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        assert isinstance(data,
                          dict), 'Please use default_collate in dataloader, \
            instead of pseudo_collate.'

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)

        # Here is what is different from :class:`mmengine.ImgDataPreprocessor`
        # Since there are multiple views for an image for some algorithms,
        # e.g. SimCLR, each item in inputs is a list, containing multi-views
        # for an image.
        if isinstance(batch_inputs, list):
            # channel transform
            if self._channel_conversion:
                batch_inputs = [
                    _input[:, [2, 1, 0], ...] for _input in batch_inputs
                ]

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = [_input.float() for _input in batch_inputs]

            # normalization.
            if self._enable_normalize:
                batch_inputs = [(_input - self.mean) / self.std
                                for _input in batch_inputs]
        else:
            # channel transform
            if self._channel_conversion:
                batch_inputs = batch_inputs[:, [2, 1, 0], ...]

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = batch_inputs.float()

            # normalization.
            if self._enable_normalize:
                batch_inputs = (batch_inputs - self.mean) / self.std

        return {'inputs': batch_inputs, 'data_samples': batch_data_samples}


@MODELS.register_module()
class TwoNormDataPreprocessor(SelfSupDataPreprocessor):
    """Image pre-processor for CAE, BEiT v1/v2, etc.

    Compared with the :class:`mmselfsup.SelfSupDataPreprocessor`, this module
    will normalize the prediction image and target image with different
    normalization parameters.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``to_rgb=True`` it means the mean value of R, G, B
            channels. If the length of `mean` is 1, it means all channels have
            the same mean value, or the input is a gray image. If it is not
            specified, images will not be normalized. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``to_rgb=True`` it means the standard deviation
            of R, G, B channels. If the length of `std` is 1, it means all
            channels have the same standard deviation, or the input is a gray
            image.  If it is not specified, images will not be normalized.
            Defaults to None.
        second_mean (Sequence[float or int], optional): The description is
            like ``mean``, it can be customized for targe image. Defaults to
            None.
        second_std (Sequence[float or int], optional): The description is
            like ``std``, it can be customized for targe image. Defaults to
            None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        non_blocking (bool): Whether block current process when transferring
            data to device. Defaults to False.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 second_mean: Sequence[Union[float, int]] = None,
                 second_std: Sequence[Union[float, int]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            to_rgb=to_rgb,
            non_blocking=non_blocking)
        assert (second_mean is not None) and (second_std is not None), (
            'mean and std should not be None while using '
            '`TwoNormDataPreprocessor`')
        assert len(second_mean) == 3 or len(second_mean) == 1, (
            '`mean` should have 1 or 3 values, to be compatible with '
            f'RGB or gray image, but got {len(second_mean)} values')
        assert len(second_std) == 3 or len(second_std) == 1, (
            '`std` should have 1 or 3 values, to be compatible with RGB '
            f'or gray image, but got {len(std)} values')

        self.register_buffer('second_mean',
                             torch.tensor(second_mean).view(-1, 1, 1), False)
        self.register_buffer('second_std',
                             torch.tensor(second_std).view(-1, 1, 1), False)

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization and bgr2rgb conversion based on
        ``BaseDataPreprocessor``. The ``batch_inputs`` in forward function is a
        list.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
                model input.
        """
        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)
        # channel transform
        if self._channel_conversion:
            batch_inputs = [
                _input[:, [2, 1, 0], ...] for _input in batch_inputs
            ]

        # convert to float after channel conversion to ensure efficiency
        batch_inputs = [_input.float() for _input in batch_inputs]

        # Normalization. Here is what is different from
        # :class:`mmselfsup.SelfSupDataPreprocessor`. Normalize the target
        # image and prediction image with different normalization params
        if self._enable_normalize:
            batch_inputs = [
                (batch_inputs[0] - self.mean) / self.std,
                (batch_inputs[1] - self.second_mean) / self.second_std
            ]

        return {'inputs': batch_inputs, 'data_samples': batch_data_samples}


@MODELS.register_module()
class VideoDataPreprocessor(BaseDataPreprocessor):
    """Video pre-processor for operations, like normalization and bgr to rgb
    conversion .

    Compared with the :class:`mmaction.ActionDataPreprocessor`, this module
    supports ``inputs`` as torch.Tensor or a list of torch.Tensor.

    Args:
        mean (Sequence[float or int, optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape == 'NCTHW':
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[List[torch.Tensor], Optional[list]]: Data in the same format
                as the model input.
        """

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)

        if isinstance(batch_inputs, list):
            # channel transform
            if self.to_rgb:
                if self.format_shape == 'NCHW':
                    batch_inputs = [
                        _input[..., [2, 1, 0], :, :] for _input in batch_inputs
                    ]
                elif self.format_shape == 'NCTHW':
                    batch_inputs = [
                        _input[..., [2, 1, 0], :, :, :]
                        for _input in batch_inputs
                    ]
                else:
                    raise ValueError(
                        f'Invalid format shape: {self.format_shape}')

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = [_input.float() for _input in batch_inputs]

            # normalization
            if self._enable_normalize:
                batch_inputs = [(_input - self.mean) / self.std
                                for _input in batch_inputs]

        else:
            # channel transform
            if self.to_rgb:
                if self.format_shape == 'NCHW':
                    batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
                elif self.format_shape == 'NCTHW':
                    batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
                else:
                    raise ValueError(
                        f'Invalid format shape: {self.format_shape}')

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = batch_inputs.float()

            # normalization
            if self._enable_normalize:
                batch_inputs = (batch_inputs - self.mean) / self.std

        return {'inputs': batch_inputs, 'data_samples': batch_data_samples}


@MODELS.register_module()
class MultiModalDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for image-text multimodality tasks.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Number = 0,
        to_rgb: bool = False,
    ):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                'preprocessing, please specify both `mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)

        imgs = data.get('inputs', None)

        def _process_img(img):
            # ------ To RGB ------
            if self.to_rgb and img.size(1) == 3:
                img = img.flip(1)

            # -- Normalization ---
            img = img.float()
            if self._enable_normalize:
                img = (img - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = img.shape[-2:]

                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                img = F.pad(img, (0, pad_w, 0, pad_h), 'constant',
                            self.pad_value)
            return img

        if isinstance(imgs, torch.Tensor):
            imgs = _process_img(imgs)
        elif isinstance(imgs, Sequence):
            # B, T, C, H, W
            imgs = torch.stack([_process_img(img) for img in imgs], dim=1)
        elif imgs is not None:
            raise ValueError(f'{type(imgs)} is not supported for imgs inputs.')

        data_samples = data.get('data_samples', None)

        return {'images': imgs, 'data_samples': data_samples}
