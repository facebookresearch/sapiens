# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
from typing import Dict, Optional

from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from .hook import DATA_BATCH, Hook


@HOOKS.register_module()
class EMAHook(Hook):
    """A Hook to apply Exponential Moving Average (EMA) on the model during
    training.

    Note:
        - EMAHook takes priority over CheckpointHook.
        - The original model parameters are actually saved in ema field after
          train.
        - ``begin_iter`` and ``begin_epoch`` cannot be set at the same time.

    Args:
        ema_type (str): The type of EMA strategy to use. You can find the
            supported strategies in :mod:`mmengine.model.averaged_model`.
            Defaults to 'ExponentialMovingAverage'.
        strict_load (bool): Whether to strictly enforce that the keys of
            ``state_dict`` in checkpoint match the keys returned by
            ``self.module.state_dict``. Defaults to False.
            Changed in v0.3.0.
        begin_iter (int): The number of iteration to enable ``EMAHook``.
            Defaults to 0.
        begin_epoch (int): The number of epoch to enable ``EMAHook``.
            Defaults to 0.
        **kwargs: Keyword arguments passed to subclasses of
            :obj:`BaseAveragedModel`
    """

    priority = 'NORMAL'

    def __init__(self,
                 ema_type: str = 'ExponentialMovingAverage',
                 strict_load: bool = False,
                 begin_iter: int = 0,
                 begin_epoch: int = 0,
                 **kwargs):
        self.strict_load = strict_load
        self.ema_cfg = dict(type=ema_type, **kwargs)
        assert not (begin_iter != 0 and begin_epoch != 0), (
            '`begin_iter` and `begin_epoch` should not be both set.')
        assert begin_iter >= 0, (
            '`begin_iter` must larger than or equal to 0, '
            f'but got begin_iter: {begin_iter}')
        assert begin_epoch >= 0, (
            '`begin_epoch` must larger than or equal to 0, '
            f'but got begin_epoch: {begin_epoch}')
        self.begin_iter = begin_iter
        self.begin_epoch = begin_epoch
        # If `begin_epoch` and `begin_iter` are not set, `EMAHook` will be
        # enabled at 0 iteration.
        self.enabled_by_epoch = self.begin_epoch > 0

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model
        self.ema_model = MODELS.build(
            self.ema_cfg, default_args=dict(model=self.src_model))

    def before_train(self, runner) -> None:
        """Check the begin_epoch/iter is smaller than max_epochs/iters.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.enabled_by_epoch:
            assert self.begin_epoch <= runner.max_epochs, (
                'self.begin_epoch should be smaller than or equal to '
                f'runner.max_epochs: {runner.max_epochs}, but got '
                f'begin_epoch: {self.begin_epoch}')
        else:
            assert self.begin_iter <= runner.max_iters, (
                'self.begin_iter should be smaller than or equal to '
                f'runner.max_iters: {runner.max_iters}, but got '
                f'begin_iter: {self.begin_iter}')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ema parameter.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self._ema_started(runner):
            self.ema_model.update_parameters(self.src_model)
        else:
            ema_params = self.ema_model.module.state_dict()
            src_params = self.src_model.state_dict()
            for k, p in ema_params.items():
                p.data.copy_(src_params[k].data)

    def before_val_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before
        validation.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._swap_ema_parameters()

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after validation.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._swap_ema_parameters()

    def before_test_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before test.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._swap_ema_parameters()

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after test.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._swap_ema_parameters()

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """Save ema parameters to checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        # Save ema parameters to the source model's state dict so that we
        # can directly load the averaged model weights for deployment.
        # Swapping the state_dict key-values instead of swapping model
        # parameters because the state_dict is a shallow copy of model
        # parameters.
        self._swap_ema_state_dict(checkpoint)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        from mmengine.runner.checkpoint import load_state_dict
        if 'ema_state_dict' in checkpoint and runner._resume:
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint['ema_state_dict'], strict=self.strict_load)

        # Support load checkpoint without ema state dict.
        else:
            if runner._resume:
                print_log(
                    'There is no `ema_state_dict` in checkpoint. '
                    '`EMAHook` will make a copy of `state_dict` as the '
                    'initial `ema_state_dict`', 'current', logging.WARNING)
            load_state_dict(
                self.ema_model.module,
                copy.deepcopy(checkpoint['state_dict']),
                strict=self.strict_load)

    def _swap_ema_parameters(self) -> None:
        """Swap the parameter of model with ema_model."""
        avg_param = (
            itertools.chain(self.ema_model.module.parameters(),
                            self.ema_model.module.buffers())
            if self.ema_model.update_buffers else
            self.ema_model.module.parameters())
        src_param = (
            itertools.chain(self.src_model.parameters(),
                            self.src_model.buffers())
            if self.ema_model.update_buffers else self.src_model.parameters())
        for p_avg, p_src in zip(avg_param, src_param):
            tmp = p_avg.data.clone()
            p_avg.data.copy_(p_src.data)
            p_src.data.copy_(tmp)

    def _swap_ema_state_dict(self, checkpoint):
        """Swap the state dict values of model with ema_model."""
        model_state = checkpoint['state_dict']
        ema_state = checkpoint['ema_state_dict']
        for k in ema_state:
            if k[:7] == 'module.':
                tmp = ema_state[k]
                ema_state[k] = model_state[k[7:]]
                model_state[k[7:]] = tmp

    def _ema_started(self, runner) -> bool:
        """Whether ``EMAHook`` has been initialized at current iteration or
        epoch.

        :attr:`ema_model` will be initialized when ``runner.iter`` or
        ``runner.epoch`` is greater than ``self.begin`` for the first time.

        Args:
            runner (Runner): Runner of the training, validation process.

        Returns:
            bool: Whether ``EMAHook`` has been initialized.
        """
        if self.enabled_by_epoch:
            return runner.epoch + 1 >= self.begin_epoch
        else:
            return runner.iter + 1 >= self.begin_iter
