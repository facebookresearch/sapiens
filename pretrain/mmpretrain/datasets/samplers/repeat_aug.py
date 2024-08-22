import math
from typing import Iterator, Optional, Sized

import torch
from mmengine.dist import get_dist_info, is_main_process, sync_random_seed
from torch.utils.data import Sampler

from mmpretrain.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class RepeatAugSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for
    distributed, with repeated augmentation. It ensures that different each
    augmented version of a sample will be visible to a different process (GPU).
    Heavily based on torch.utils.data.DistributedSampler.

    This sampler was taken from
    https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        num_repeats (int): The repeat times of every sample. Defaults to 3.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 num_repeats: int = 3,
                 seed: Optional[int] = None):
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if not self.shuffle and is_main_process():
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.warning('The RepeatAugSampler always picks a '
                           'fixed part of data if `shuffle=False`.')

        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.num_repeats = num_repeats

        # The number of repeated samples in the rank
        self.num_samples = math.ceil(
            len(self.dataset) * num_repeats / world_size)
        # The total number of repeated samples in all ranks.
        self.total_size = self.num_samples * world_size
        # The number of selected samples in the rank
        self.num_selected_samples = math.ceil(len(self.dataset) / world_size)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        indices = [x for x in indices for _ in range(self.num_repeats)]
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_selected_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
