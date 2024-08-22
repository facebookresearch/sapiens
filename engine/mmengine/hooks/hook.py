# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Sequence, Union

from mmengine import is_method_overridden

DATA_BATCH = Optional[Union[dict, tuple, list]]


class Hook:
    """Base hook class.

    All hooks should inherit from this class.
    """

    priority = 'NORMAL'
    stages = ('before_run', 'after_load_checkpoint', 'before_train',
              'before_train_epoch', 'before_train_iter', 'after_train_iter',
              'after_train_epoch', 'before_val', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_val', 'before_save_checkpoint', 'after_train',
              'before_test', 'before_test_epoch', 'before_test_iter',
              'after_test_iter', 'after_test_epoch', 'after_test', 'after_run')

    def before_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """

    def after_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """

    def before_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before train.

        Args:
            runner (Runner): The runner of the training process.
        """

    def after_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after train.

        Args:
            runner (Runner): The runner of the training process.
        """

    def before_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before validation.

        Args:
            runner (Runner): The runner of the validation process.
        """

    def after_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after validation.

        Args:
            runner (Runner): The runner of the validation process.
        """

    def before_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before testing.

        Args:
            runner (Runner): The runner of the testing process.
        """

    def after_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after testing.

        Args:
            runner (Runner): The runner of the testing process.
        """

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations before saving the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations after loading the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """

    def before_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._before_epoch(runner, mode='train')

    def before_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
        """
        self._before_epoch(runner, mode='val')

    def before_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
        """
        self._before_epoch(runner, mode='test')

    def after_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._after_epoch(runner, mode='train')

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._after_epoch(runner, mode='val')

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._after_epoch(runner, mode='test')

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='train')

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='val')

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each test iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='test')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='train')

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='val')

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test iteration.

        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='test')

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def _after_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each iter.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[Sequence, dict]] = None,
                    mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict or Sequence, optional): Outputs from model.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def every_n_epochs(self, runner, n: int, start: int = 0) -> bool:
        """Test whether current epoch can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current epoch can be evenly divided by n.
            start (int): Starting from `start` to check the logic for
                every n epochs. Defaults to 0.

        Returns:
            bool: Whether current epoch can be evenly divided by n.
        """
        dividend = runner.epoch + 1 - start
        return dividend % n == 0 if dividend >= 0 and n > 0 else False

    def every_n_inner_iters(self, batch_idx: int, n: int) -> bool:
        """Test whether current inner iteration can be evenly divided by n.

        Args:
            batch_idx (int): Current batch index of the training, validation
                or testing loop.
            n (int): Whether current inner iteration can be evenly
                divided by n.

        Returns:
            bool: Whether current inner iteration can be evenly
            divided by n.
        """
        return (batch_idx + 1) % n == 0 if n > 0 else False

    def every_n_train_iters(self, runner, n: int, start: int = 0) -> bool:
        """Test whether current training iteration can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current iteration can be evenly divided by n.
            start (int): Starting from `start` to check the logic for
                every n iterations. Defaults to 0.

        Returns:
            bool: Return True if the current iteration can be evenly divided
            by n, otherwise False.
        """
        dividend = runner.iter + 1 - start
        return dividend % n == 0 if dividend >= 0 and n > 0 else False

    def end_of_epoch(self, dataloader, batch_idx: int) -> bool:
        """Check whether the current iteration reaches the last iteration of
        the dataloader.

        Args:
            dataloader (Dataloader): The dataloader of the training,
                validation or testing process.
            batch_idx (int): The index of the current batch in the loop.
        Returns:
            bool: Whether reaches the end of current epoch or not.
        """
        return batch_idx + 1 == len(dataloader)

    def is_last_train_epoch(self, runner) -> bool:
        """Test whether current epoch is the last train epoch.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: Whether reaches the end of training epoch.
        """
        return runner.epoch + 1 == runner.max_epochs

    def is_last_train_iter(self, runner) -> bool:
        """Test whether current iteration is the last train iteration.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: Whether current iteration is the last train iteration.
        """
        return runner.iter + 1 == runner.max_iters

    def get_triggered_stages(self) -> list:
        """Get all triggered stages with method name of the hook.

        Returns:
            list: List of triggered stages.
        """
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)

        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            '_before_epoch':
            ['before_train_epoch', 'before_val_epoch', 'before_test_epoch'],
            '_after_epoch':
            ['after_train_epoch', 'after_val_epoch', 'after_test_epoch'],
            '_before_iter':
            ['before_train_iter', 'before_val_iter', 'before_test_iter'],
            '_after_iter':
            ['after_train_iter', 'after_val_iter', 'after_test_iter'],
        }

        for method, map_stages in method_stages_map.items():
            if is_method_overridden(method, Hook, self):
                trigger_stages.update(map_stages)

        return list(trigger_stages)
