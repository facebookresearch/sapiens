# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

from mmengine.hooks import Hook

from mmpretrain.registry import HOOKS
from mmpretrain.utils import get_ori_model


@HOOKS.register_module()
class DenseCLHook(Hook):
    """Hook for DenseCL.

    This hook includes ``loss_lambda`` warmup in DenseCL.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL>`_.

    Args:
        start_iters (int): The number of warmup iterations to set
            ``loss_lambda=0``. Defaults to 1000.
    """

    def __init__(self, start_iters: int = 1000) -> None:
        self.start_iters = start_iters

    def before_train(self, runner) -> None:
        """Obtain ``loss_lambda`` from algorithm."""
        assert hasattr(get_ori_model(runner.model), 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCL."
        self.loss_lambda = get_ori_model(runner.model).loss_lambda

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        """Adjust ``loss_lambda`` every train iter."""
        assert hasattr(get_ori_model(runner.model), 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCL."
        cur_iter = runner.iter
        if cur_iter >= self.start_iters:
            get_ori_model(runner.model).loss_lambda = self.loss_lambda
        else:
            get_ori_model(runner.model).loss_lambda = 0.
