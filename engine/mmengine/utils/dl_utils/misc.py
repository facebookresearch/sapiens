# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pkgutil
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..misc import is_tuple_of
from .parrots_wrapper import _BatchNorm, _InstanceNorm


def is_norm(layer: nn.Module,
            exclude: Optional[Union[type, Tuple[type]]] = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type, tuple[type], optional): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)


def tensor2imgs(tensor: torch.Tensor,
                mean: Optional[Tuple[float, float, float]] = None,
                std: Optional[Tuple[float, float, float]] = None,
                to_bgr: bool = True):
    """Convert tensor to 3-channel images or 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W). :math:`C` can be either 3 or 1. If C is 3, the format
            should be RGB.
        mean (tuple[float], optional): Mean of images. If None,
            (0, 0, 0) will be used for tensor with 3-channel,
            while (0, ) for tensor with 1-channel. Defaults to None.
        std (tuple[float], optional): Standard deviation of images. If None,
            (1, 1, 1) will be used for tensor with 3-channel,
            while (1, ) for tensor with 1-channel. Defaults to None.
        to_bgr (bool): For the tensor with 3 channel, convert its format to
            BGR. For the tensor with 1 channel, it must be False. Defaults to
            True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    assert torch.is_tensor(tensor) and tensor.ndim == 4
    channels = tensor.size(1)
    assert channels in [1, 3]
    if mean is None:
        mean = (0, ) * channels
    if std is None:
        std = (1, ) * channels
    assert (channels == len(mean) == len(std) == 3) or \
           (channels == len(mean) == len(std) == 1 and not to_bgr)
    mean = tensor.new_tensor(mean).view(1, -1)
    std = tensor.new_tensor(std).view(1, -1)
    tensor = tensor.permute(0, 2, 3, 1) * std + mean
    imgs = tensor.detach().cpu().numpy()
    if to_bgr and channels == 3:
        imgs = imgs[:, :, :, (2, 1, 0)]  # RGB2BGR
    imgs = [np.ascontiguousarray(img) for img in imgs]
    return imgs


def has_batch_norm(model: nn.Module) -> bool:
    """Detect whether model has a BatchNormalization layer.

    Args:
        model (nn.Module): training model.

    Returns:
        bool: whether model has a BatchNormalization layer
    """
    if isinstance(model, _BatchNorm):
        return True
    for m in model.children():
        if has_batch_norm(m):
            return True
    return False


def mmcv_full_available() -> bool:
    """Check whether mmcv-full is installed.

    Returns:
        bool: True if mmcv-full is installed else False.
    """
    try:
        import mmcv  # noqa: F401
    except ImportError:
        return False
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None
