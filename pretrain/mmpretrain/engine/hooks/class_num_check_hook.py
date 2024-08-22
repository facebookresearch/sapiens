# Copyright (c) OpenMMLab. All rights reserved
from mmengine.hooks import Hook
from mmengine.utils import is_seq_of

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class ClassNumCheckHook(Hook):
    """Class Number Check HOOK."""

    def _check_head(self, runner, dataset):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`Runner`): runner object.
            dataset (obj: `BaseDataset`): the dataset to check.
        """
        model = runner.model
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set class information in `metainfo` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        else:
            assert is_seq_of(dataset.CLASSES, str), \
                (f'Class information in `metainfo` in '
                 f'{dataset.__class__.__name__} should be a tuple of str.')
            for _, module in model.named_modules():
                if hasattr(module, 'num_classes'):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of class information in `metainfo` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')

    def before_train(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj: `IterBasedRunner`): Iter based Runner.
        """
        self._check_head(runner, runner.train_dataloader.dataset)

    def before_val(self, runner):
        """Check whether the validation dataset is compatible with head.

        Args:
            runner (obj:`IterBasedRunner`): Iter based Runner.
        """
        self._check_head(runner, runner.val_dataloader.dataset)

    def before_test(self, runner):
        """Check whether the test dataset is compatible with head.

        Args:
            runner (obj:`IterBasedRunner`): Iter based Runner.
        """
        self._check_head(runner, runner.test_dataloader.dataset)
