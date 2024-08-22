# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union

import numpy as np
import torch

from mmpretrain.registry import BATCH_AUGMENTS


class RandomBatchAugment:
    """Randomly choose one batch augmentation to apply.

    Args:
        augments (Callable | dict | list): configs of batch
            augmentations.
        probs (float | List[float] | None): The probabilities of each batch
            augmentations. If None, choose evenly. Defaults to None.

    Example:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from mmpretrain.models import RandomBatchAugment
        >>> augments_cfg = [
        ...     dict(type='CutMix', alpha=1.),
        ...     dict(type='Mixup', alpha=1.)
        ... ]
        >>> batch_augment = RandomBatchAugment(augments_cfg, probs=[0.5, 0.3])
        >>> imgs = torch.rand(16, 3, 32, 32)
        >>> label = F.one_hot(torch.randint(0, 10, (16, )), num_classes=10)
        >>> imgs, label = batch_augment(imgs, label)

    .. note ::

        To decide which batch augmentation will be used, it picks one of
        ``augments`` based on the probabilities. In the example above, the
        probability to use CutMix is 0.5, to use Mixup is 0.3, and to do
        nothing is 0.2.
    """

    def __init__(self, augments: Union[Callable, dict, list], probs=None):
        if not isinstance(augments, (tuple, list)):
            augments = [augments]

        self.augments = []
        for aug in augments:
            if isinstance(aug, dict):
                self.augments.append(BATCH_AUGMENTS.build(aug))
            else:
                self.augments.append(aug)

        if isinstance(probs, float):
            probs = [probs]

        if probs is not None:
            assert len(augments) == len(probs), \
                '``augments`` and ``probs`` must have same lengths. ' \
                f'Got {len(augments)} vs {len(probs)}.'
            assert sum(probs) <= 1, \
                'The total probability of batch augments exceeds 1.'
            self.augments.append(None)
            probs.append(1 - sum(probs))

        self.probs = probs

    def __call__(self, batch_input: torch.Tensor, batch_score: torch.Tensor):
        """Randomly apply batch augmentations to the batch inputs and batch
        data samples."""
        aug_index = np.random.choice(len(self.augments), p=self.probs)
        aug = self.augments[aug_index]

        if aug is not None:
            return aug(batch_input, batch_score)
        else:
            return batch_input, batch_score.float()
