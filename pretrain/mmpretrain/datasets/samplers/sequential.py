# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

import torch
from mmengine.dataset import DefaultSampler

from mmpretrain.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SequentialSampler(DefaultSampler):
    """Sequential sampler which supports different subsample policy.

    Args:
        dataset (Sized): The dataset.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
        subsample_type (str): The method to subsample data on different rank.
            Supported type:

            - ``'default'``: Original torch behavior. Sample the examples one
              by one for each GPU in terms. For instance, 8 examples on 2 GPUs,
              GPU0: [0,2,4,8], GPU1: [1,3,5,7]
            - ``'sequential'``: Subsample all examples to n chunk sequntially.
              For instance, 8 examples on 2 GPUs,
              GPU0: [0,1,2,3], GPU1: [4,5,6,7]
    """

    def __init__(self, subsample_type: str = 'default', **kwargs) -> None:
        super().__init__(shuffle=False, **kwargs)

        if subsample_type not in ['default', 'sequential']:
            raise ValueError(f'Unsupported subsample typer "{subsample_type}",'
                             ' please choose from ["default", "sequential"]')
        self.subsample_type = subsample_type

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        if self.subsample_type == 'default':
            indices = indices[self.rank:self.total_size:self.world_size]
        elif self.subsample_type == 'sequential':
            num_samples_per_rank = self.total_size // self.world_size
            indices = indices[self.rank *
                              num_samples_per_rank:(self.rank + 1) *
                              num_samples_per_rank]

        return iter(indices)
