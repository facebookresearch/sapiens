# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor


@MODELS.register_module()
class SwAV(BaseSelfSupervisor):
    """SwAV.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_.

    The queue is built in ``mmpretrain/engine/hooks/swav_hook.py``.
    """

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        # multi-res forward passes
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([input.shape[-1] for input in inputs]),
                return_counts=True)[1], 0)
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(inputs[start_idx:end_idx]))
            output.append(_out)
            start_idx = end_idx
        output = self.neck(output)

        loss = self.head.loss(output)
        losses = dict(loss=loss)
        return losses
