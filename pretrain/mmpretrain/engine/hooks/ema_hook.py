# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import warnings
from typing import Dict, Optional

from mmengine.hooks import EMAHook as BaseEMAHook
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class EMAHook(BaseEMAHook):
    """A Hook to apply Exponential Moving Average (EMA) on the model during
    training.

    Comparing with :class:`mmengine.hooks.EMAHook`, this hook accepts
    ``evaluate_on_ema`` and ``evaluate_on_origin`` arguments. By default, the
    ``evaluate_on_ema`` is enabled, and if you want to do validation and
    testing on both original and EMA models, please set both arguments
    ``True``.

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
        evaluate_on_ema (bool): Whether to evaluate (validate and test)
            on EMA model during val-loop and test-loop. Defaults to True.
        evaluate_on_origin (bool): Whether to evaluate (validate and test)
            on the original model during val-loop and test-loop.
            Defaults to False.
        **kwargs: Keyword arguments passed to subclasses of
            :obj:`BaseAveragedModel`
    """

    priority = 'NORMAL'

    def __init__(self,
                 ema_type: str = 'ExponentialMovingAverage',
                 strict_load: bool = False,
                 begin_iter: int = 0,
                 begin_epoch: int = 0,
                 evaluate_on_ema: bool = True,
                 evaluate_on_origin: bool = False,
                 **kwargs):
        super().__init__(
            ema_type=ema_type,
            strict_load=strict_load,
            begin_iter=begin_iter,
            begin_epoch=begin_epoch,
            **kwargs)

        if not evaluate_on_ema and not evaluate_on_origin:
            warnings.warn(
                'Automatically set `evaluate_on_origin=True` since the '
                '`evaluate_on_ema` is disabled. If you want to disable '
                'all validation, please modify the `val_interval` of '
                'the `train_cfg`.', UserWarning)
            evaluate_on_origin = True

        self.evaluate_on_ema = evaluate_on_ema
        self.evaluate_on_origin = evaluate_on_origin
        self.load_ema_from_ckpt = False

    def before_train(self, runner) -> None:
        super().before_train(runner)
        if not runner._resume and self.load_ema_from_ckpt:
            # If loaded EMA state dict but not want to resume training
            # overwrite the EMA state dict with the source model.
            MMLogger.get_current_instance().info(
                'Load from a checkpoint with EMA parameters but not '
                'resume training. Initialize the model parameters with '
                'EMA parameters')
            for p_ema, p_src in zip(self._ema_params, self._src_params):
                p_src.data.copy_(p_ema.data)

    def before_val_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before
        validation.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.evaluate_on_ema:
            # Swap when evaluate on ema
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
        if self.evaluate_on_ema:
            # Swap when evaluate on ema
            self._swap_ema_parameters()

        if self.evaluate_on_ema and self.evaluate_on_origin:
            # Re-evaluate if evaluate on both ema and origin.
            val_loop = runner.val_loop

            runner.model.eval()
            for idx, data_batch in enumerate(val_loop.dataloader):
                val_loop.run_iter(idx, data_batch)

            # compute metrics
            origin_metrics = val_loop.evaluator.evaluate(
                len(val_loop.dataloader.dataset))

            for k, v in origin_metrics.items():
                runner.message_hub.update_scalar(f'val/{k}_origin', v)

    def before_test_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before test.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.evaluate_on_ema:
            # Swap when evaluate on ema
            self._swap_ema_parameters()
            MMLogger.get_current_instance().info('Start testing on EMA model.')
        else:
            MMLogger.get_current_instance().info(
                'Start testing on the original model.')

    def after_test_epoch(self,
                         runner: Runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after test.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        if self.evaluate_on_ema:
            # Swap when evaluate on ema
            self._swap_ema_parameters()

        if self.evaluate_on_ema and self.evaluate_on_origin:
            # Re-evaluate if evaluate on both ema and origin.
            MMLogger.get_current_instance().info(
                'Start testing on the original model.')
            test_loop = runner.test_loop

            runner.model.eval()
            for idx, data_batch in enumerate(test_loop.dataloader):
                test_loop.run_iter(idx, data_batch)

            # compute metrics
            origin_metrics = test_loop.evaluator.evaluate(
                len(test_loop.dataloader.dataset))

            for k, v in origin_metrics.items():
                runner.message_hub.update_scalar(f'test/{k}_origin', v)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        from mmengine.runner.checkpoint import load_state_dict
        if 'ema_state_dict' in checkpoint:
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint['ema_state_dict'], strict=self.strict_load)
            self.load_ema_from_ckpt = True

        # Support load checkpoint without ema state dict.
        else:
            load_state_dict(
                self.ema_model.module,
                copy.deepcopy(checkpoint['state_dict']),
                strict=self.strict_load)

    @property
    def _src_params(self):
        if self.ema_model.update_buffers:
            return itertools.chain(self.src_model.parameters(),
                                   self.src_model.buffers())
        else:
            return self.src_model.parameters()

    @property
    def _ema_params(self):
        if self.ema_model.update_buffers:
            return itertools.chain(self.ema_model.module.parameters(),
                                   self.ema_model.module.buffers())
        else:
            return self.ema_model.module.parameters()
