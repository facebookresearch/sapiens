# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
import torch

from mmpretrain.registry import BATCH_AUGMENTS
from .mixup import Mixup


@BATCH_AUGMENTS.register_module()
class CutMix(Mixup):
    r"""CutMix batch agumentation.

    CutMix is a method to improve the network's generalization capability. It's
    proposed in `CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features <https://arxiv.org/abs/1905.04899>`

    With this method, patches are cut and pasted among training images where
    the ground truth labels are also mixed proportionally to the area of the
    patches.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            can be found in :class:`Mixup`.
        cutmix_minmax (List[float], optional): The min/max area ratio of the
            patches. If not None, the bounding-box of patches is uniform
            sampled within this ratio range, and the ``alpha`` will be ignored.
            Otherwise, the bounding-box is generated according to the
            ``alpha``. Defaults to None.
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Defaults to True.

    .. note ::
        If the ``cutmix_minmax`` is None, how to generate the bounding-box of
        patches according to the ``alpha``?

        First, generate a :math:`\lambda`, details can be found in
        :class:`Mixup`. And then, the area ratio of the bounding-box
        is calculated by:

        .. math::
            \text{ratio} = \sqrt{1-\lambda}
    """

    def __init__(self,
                 alpha: float,
                 cutmix_minmax: Optional[List[float]] = None,
                 correct_lam: bool = True):
        super().__init__(alpha=alpha)

        self.cutmix_minmax = cutmix_minmax
        self.correct_lam = correct_lam

    def rand_bbox_minmax(
            self,
            img_shape: Tuple[int, int],
            count: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Min-Max CutMix bounding-box Inspired by Darknet cutmix
        implementation. It generates a random rectangular bbox based on min/max
        percent values applied to each dimension of the input image.

        Typical defaults for minmax are usually in the  .2-.3 for min and
        .8-.9 range for max.

        Args:
            img_shape (tuple): Image shape as tuple
            count (int, optional): Number of bbox to generate. Defaults to None
        """
        assert len(self.cutmix_minmax) == 2
        img_h, img_w = img_shape
        cut_h = np.random.randint(
            int(img_h * self.cutmix_minmax[0]),
            int(img_h * self.cutmix_minmax[1]),
            size=count)
        cut_w = np.random.randint(
            int(img_w * self.cutmix_minmax[0]),
            int(img_w * self.cutmix_minmax[1]),
            size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    def rand_bbox(self,
                  img_shape: Tuple[int, int],
                  lam: float,
                  margin: float = 0.,
                  count: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Standard CutMix bounding-box that generates a random square bbox
        based on lambda value. This implementation includes support for
        enforcing a border margin as percent of bbox dimensions.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            margin (float): Percentage of bbox dimension to enforce as margin
                (reduce amount of box outside image). Defaults to 0.
            count (int, optional): Number of bbox to generate. Defaults to None
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    def cutmix_bbox_and_lam(self,
                            img_shape: Tuple[int, int],
                            lam: float,
                            count: Optional[int] = None) -> tuple:
        """Generate bbox and apply lambda correction.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            count (int, optional): Number of bbox to generate. Defaults to None
        """
        if self.cutmix_minmax is not None:
            yl, yu, xl, xu = self.rand_bbox_minmax(img_shape, count=count)
        else:
            yl, yu, xl, xu = self.rand_bbox(img_shape, lam, count=count)
        if self.correct_lam or self.cutmix_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[0] * img_shape[1])
        return (yl, yu, xl, xu), lam

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
        batch_size = batch_inputs.size(0)
        img_shape = batch_inputs.shape[-2:]
        index = torch.randperm(batch_size)

        (y1, y2, x1, x2), lam = self.cutmix_bbox_and_lam(img_shape, lam)
        batch_inputs[:, :, y1:y2, x1:x2] = batch_inputs[index, :, y1:y2, x1:x2]
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return batch_inputs, mixed_scores
