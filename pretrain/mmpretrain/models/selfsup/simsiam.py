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
class SimSiam(BaseSelfSupervisor):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_. The operation of fixing learning rate
    of predictor is in `engine/hooks/simsiam_hook.py`.
    """

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(inputs, list)
        img_v1 = inputs[0]
        img_v2 = inputs[1]

        z1 = self.neck(self.backbone(img_v1))[0]  # NxC
        z2 = self.neck(self.backbone(img_v2))[0]  # NxC

        loss_1 = self.head.loss(z1, z2)
        loss_2 = self.head.loss(z2, z1)

        losses = dict(loss=0.5 * (loss_1 + loss_2))
        return losses
