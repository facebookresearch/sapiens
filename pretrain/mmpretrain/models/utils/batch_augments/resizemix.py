# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mmpretrain.registry import BATCH_AUGMENTS
from .cutmix import CutMix


@BATCH_AUGMENTS.register_module()
class ResizeMix(CutMix):
    r"""ResizeMix Random Paste layer for a batch of data.

    The ResizeMix will resize an image to a small patch and paste it on another
    image. It's proposed in `ResizeMix: Mixing Data with Preserved Object
    Information and True Labels <https://arxiv.org/abs/2012.11101>`_

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            can be found in :class:`Mixup`.
        lam_min(float): The minimum value of lam. Defaults to 0.1.
        lam_max(float): The maximum value of lam. Defaults to 0.8.
        interpolation (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' |
            'area'. Defaults to 'bilinear'.
        prob (float): The probability to execute resizemix. It should be in
            range [0, 1]. Defaults to 1.0.
        cutmix_minmax (List[float], optional): The min/max area ratio of the
            patches. If not None, the bounding-box of patches is uniform
            sampled within this ratio range, and the ``alpha`` will be ignored.
            Otherwise, the bounding-box is generated according to the
            ``alpha``. Defaults to None.
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Defaults to True
        **kwargs: Any other parameters accpeted by :class:`CutMix`.

    Note:
        The :math:`\lambda` (``lam``) is the mixing ratio. It's a random
        variable which follows :math:`Beta(\alpha, \alpha)` and is mapped
        to the range [``lam_min``, ``lam_max``].

        .. math::
            \lambda = \frac{Beta(\alpha, \alpha)}
            {\lambda_{max} - \lambda_{min}} + \lambda_{min}

        And the resize ratio of source images is calculated by :math:`\lambda`:

        .. math::
            \text{ratio} = \sqrt{1-\lambda}
    """

    def __init__(self,
                 alpha: float,
                 lam_min: float = 0.1,
                 lam_max: float = 0.8,
                 interpolation: str = 'bilinear',
                 cutmix_minmax: Optional[List[float]] = None,
                 correct_lam: bool = True):
        super().__init__(
            alpha=alpha, cutmix_minmax=cutmix_minmax, correct_lam=correct_lam)
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.interpolation = interpolation

    def mix(self, batch_inputs: torch.Tensor,
            batch_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix the batch inputs and batch one-hot format ground truth.

        Args:
            batch_inputs (Tensor): A batch of images tensor in the shape of
                ``(N, C, H, W)``.
            batch_scores (Tensor): A batch of one-hot format labels in the
                shape of ``(N, num_classes)``.

        Returns:
            Tuple[Tensor, Tensor): The mixed inputs and labels.
        """
        lam = np.random.beta(self.alpha, self.alpha)
        lam = lam * (self.lam_max - self.lam_min) + self.lam_min
        img_shape = batch_inputs.shape[-2:]
        batch_size = batch_inputs.size(0)
        index = torch.randperm(batch_size)

        (y1, y2, x1, x2), lam = self.cutmix_bbox_and_lam(img_shape, lam)
        batch_inputs[:, :, y1:y2, x1:x2] = F.interpolate(
            batch_inputs[index],
            size=(y2 - y1, x2 - x1),
            mode=self.interpolation,
            align_corners=False)
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return batch_inputs, mixed_scores
