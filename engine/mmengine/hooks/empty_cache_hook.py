# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Union

import torch

from mmengine.registry import HOOKS
from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class EmptyCacheHook(Hook):
    """Releases all unoccupied cached GPU memory during the process of
    training.

    Args:
        before_epoch (bool): Whether to release cache before an epoch. Defaults
            to False.
        after_epoch (bool): Whether to release cache after an epoch. Defaults
            to True.
        after_iter (bool): Whether to release cache after an iteration.
            Defaults to False.
    """

    priority = 'NORMAL'

    def __init__(self,
                 before_epoch: bool = False,
                 after_epoch: bool = True,
                 after_iter: bool = False) -> None:
        self._do_before_epoch = before_epoch
        self._do_after_epoch = after_epoch
        self._do_after_iter = after_iter

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[dict, Sequence]] = None,
                    mode: str = 'train') -> None:
        """Empty cache after an iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict or sequence, optional): Outputs from model.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if self._do_after_iter:
            torch.cuda.empty_cache()

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """Empty cache before an epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if self._do_before_epoch:
            torch.cuda.empty_cache()

    def _after_epoch(self, runner, mode: str = 'train') -> None:
        """Empty cache after an epoch.

        Args:
            runner (Runner): The runner of the training process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """
        if self._do_after_epoch:
            torch.cuda.empty_cache()
