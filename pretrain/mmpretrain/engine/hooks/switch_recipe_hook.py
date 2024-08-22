# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from copy import deepcopy

from mmcv.transforms import Compose
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmpretrain.models.utils import RandomBatchAugment
from mmpretrain.registry import HOOKS, MODEL_WRAPPERS, MODELS


@HOOKS.register_module()
class SwitchRecipeHook(Hook):
    """switch recipe during the training loop, including train pipeline, batch
    augments and loss currently.

    Args:
        schedule (list): Every item of the schedule list should be a dict, and
            the dict should have ``action_epoch`` and some of
            ``train_pipeline``, ``train_augments`` and ``loss`` keys:

            - ``action_epoch`` (int): switch training recipe at which epoch.
            - ``train_pipeline`` (list, optional): The new data pipeline of the
              train dataset. If not specified, keep the original settings.
            - ``batch_augments`` (dict | None, optional): The new batch
              augmentations of during training. See :mod:`Batch Augmentations
              <mmpretrain.models.utils.batch_augments>` for more details.
              If None, disable batch augmentations. If not specified, keep the
              original settings.
            - ``loss`` (dict, optional): The new loss module config. If not
              specified, keep the original settings.

    Example:
        To use this hook in config files.

        .. code:: python

            custom_hooks = [
                dict(
                    type='SwitchRecipeHook',
                    schedule=[
                        dict(
                            action_epoch=30,
                            train_pipeline=pipeline_after_30e,
                            batch_augments=batch_augments_after_30e,
                            loss=loss_after_30e,
                        ),
                        dict(
                            action_epoch=60,
                            # Disable batch augmentations after 60e
                            # and keep other settings.
                            batch_augments=None,
                        ),
                    ]
                )
            ]
    """
    priority = 'NORMAL'

    def __init__(self, schedule):
        recipes = {}
        for recipe in schedule:
            assert 'action_epoch' in recipe, \
                'Please set `action_epoch` in every item ' \
                'of the `schedule` in the SwitchRecipeHook.'
            recipe = deepcopy(recipe)
            if 'train_pipeline' in recipe:
                recipe['train_pipeline'] = Compose(recipe['train_pipeline'])
            if 'batch_augments' in recipe:
                batch_augments = recipe['batch_augments']
                if isinstance(batch_augments, dict):
                    batch_augments = RandomBatchAugment(**batch_augments)
                recipe['batch_augments'] = batch_augments
            if 'loss' in recipe:
                loss = recipe['loss']
                if isinstance(loss, dict):
                    loss = MODELS.build(loss)
                recipe['loss'] = loss

            action_epoch = recipe.pop('action_epoch')
            assert action_epoch not in recipes, \
                f'The `action_epoch` {action_epoch} is repeated ' \
                'in the SwitchRecipeHook.'
            recipes[action_epoch] = recipe
        self.schedule = OrderedDict(sorted(recipes.items()))

    def before_train(self, runner) -> None:
        """before run setting. If resume form a checkpoint, do all switch
        before the current epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """
        if runner._resume:
            for action_epoch, recipe in self.schedule.items():
                if action_epoch >= runner.epoch + 1:
                    break
                self._do_switch(runner, recipe,
                                f' (resume recipe of epoch {action_epoch})')

    def before_train_epoch(self, runner):
        """do before train epoch."""
        recipe = self.schedule.get(runner.epoch + 1, None)
        if recipe is not None:
            self._do_switch(runner, recipe, f' at epoch {runner.epoch + 1}')

    def _do_switch(self, runner, recipe, extra_info=''):
        """do the switch aug process."""
        if 'batch_augments' in recipe:
            self._switch_batch_augments(runner, recipe['batch_augments'])
            runner.logger.info(f'Switch batch augments{extra_info}.')

        if 'train_pipeline' in recipe:
            self._switch_train_pipeline(runner, recipe['train_pipeline'])
            runner.logger.info(f'Switch train pipeline{extra_info}.')

        if 'loss' in recipe:
            self._switch_loss(runner, recipe['loss'])
            runner.logger.info(f'Switch loss{extra_info}.')

    @staticmethod
    def _switch_batch_augments(runner, batch_augments):
        """switch the train augments."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        model.data_preprocessor.batch_augments = batch_augments

    @staticmethod
    def _switch_train_pipeline(runner, train_pipeline):
        """switch the train loader dataset pipeline."""

        def switch_pipeline(dataset, pipeline):
            if hasattr(dataset, 'pipeline'):
                # for usual dataset
                dataset.pipeline = pipeline
            elif hasattr(dataset, 'datasets'):
                # for concat dataset wrapper
                for ds in dataset.datasets:
                    switch_pipeline(ds, pipeline)
            elif hasattr(dataset, 'dataset'):
                # for other dataset wrappers
                switch_pipeline(dataset.dataset, pipeline)
            else:
                raise RuntimeError(
                    'Cannot access the `pipeline` of the dataset.')

        train_loader = runner.train_loop.dataloader
        switch_pipeline(train_loader.dataset, train_pipeline)

        # To restart the iterator of dataloader when `persistent_workers=True`
        train_loader._iterator = None

    @staticmethod
    def _switch_loss(runner, loss_module):
        """switch the loss module."""
        model = runner.model
        if is_model_wrapper(model, MODEL_WRAPPERS):
            model = model.module

        if hasattr(model, 'loss_module'):
            model.loss_module = loss_module
        elif hasattr(model, 'head') and hasattr(model.head, 'loss_module'):
            model.head.loss_module = loss_module
        else:
            raise RuntimeError('Cannot access the `loss_module` of the model.')
