# Copyright (c) OpenMMLab. All rights reserved
import numpy as np
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmpretrain.models.heads import ArcFaceClsHead
from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class SetAdaptiveMarginsHook(Hook):
    r"""Set adaptive-margins in ArcFaceClsHead based on the power of
    category-wise count.

    A PyTorch implementation of paper `Google Landmark Recognition 2020
    Competition Third Place Solution <https://arxiv.org/abs/2010.05350>`_.
    The margins will be
    :math:`\text{f}(n) = (marginMax - marginMin) · norm(n^p) + marginMin`.
    The `n` indicates the number of occurrences of a category.

    Args:
        margin_min (float): Lower bound of margins. Defaults to 0.05.
        margin_max (float): Upper bound of margins. Defaults to 0.5.
        power (float): The power of category freqercy. Defaults to -0.25.
    """

    def __init__(self, margin_min=0.05, margin_max=0.5, power=-0.25) -> None:
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.margin_range = margin_max - margin_min
        self.p = power

    def before_train(self, runner):
        """change the margins in ArcFaceClsHead.

        Args:
            runner (obj: `Runner`): Runner.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if (hasattr(model, 'head')
                and not isinstance(model.head, ArcFaceClsHead)):
            raise ValueError(
                'Hook ``SetFreqPowAdvMarginsHook`` could only be used '
                f'for ``ArcFaceClsHead``, but get {type(model.head)}')

        # generate margins base on the dataset.
        gt_labels = runner.train_dataloader.dataset.get_gt_labels()
        label_count = np.bincount(gt_labels)
        label_count[label_count == 0] = 1  # At least one occurrence
        pow_freq = np.power(label_count, self.p)

        min_f, max_f = pow_freq.min(), pow_freq.max()
        normized_pow_freq = (pow_freq - min_f) / (max_f - min_f)
        margins = normized_pow_freq * self.margin_range + self.margin_min

        assert len(margins) == runner.model.head.num_classes

        model.head.set_margins(margins)
