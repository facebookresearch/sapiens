# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import torch

from mmpretrain.registry import BATCH_AUGMENTS


@BATCH_AUGMENTS.register_module()
class Mixup:
    r"""Mixup batch augmentation.

    Mixup is a method to reduces the memorization of corrupt labels and
    increases the robustness to adversarial examples. It's proposed in
    `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            are in the note.

    Note:
        The :math:`\alpha` (``alpha``) determines a random distribution
        :math:`Beta(\alpha, \alpha)`. For each batch of data, we sample
        a mixing ratio (marked as :math:`\lambda`, ``lam``) from the random
        distribution.
    """

    def __init__(self, alpha: float):
        assert isinstance(alpha, float) and alpha > 0

        self.alpha = alpha

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
        index = torch.randperm(batch_size)

        mixed_inputs = lam * batch_inputs + (1 - lam) * batch_inputs[index, :]
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return mixed_inputs, mixed_scores

    def __call__(self, batch_inputs: torch.Tensor, batch_score: torch.Tensor):
        """Mix the batch inputs and batch data samples."""
        assert batch_score.ndim == 2, \
            'The input `batch_score` should be a one-hot format tensor, '\
            'which shape should be ``(N, num_classes)``.'

        mixed_inputs, mixed_score = self.mix(batch_inputs, batch_score.float())
        return mixed_inputs, mixed_score
