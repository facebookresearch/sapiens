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
class EVA(BaseSelfSupervisor):
    """EVA.

    Implementation of `EVA: Exploring the Limits of Masked Visual
    Representation Learning at Scale <https://arxiv.org/abs/2211.07636>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        clip_feature, _ = self.target_generator(inputs)

        latent, mask, ids_restore = self.backbone(inputs)
        pred = self.neck(latent, ids_restore)

        clip_feature = clip_feature[:, 1:, :]
        loss = self.head.loss(pred, clip_feature, mask)
        losses = dict(loss=loss)
        return losses
