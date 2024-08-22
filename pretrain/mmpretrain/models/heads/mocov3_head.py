# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MoCoV3Head(BaseModule):
    """Head for MoCo v3 Pre-training.

    This head builds a predictor, which can be any registered neck component.
    It also implements latent contrastive loss between two forward features.
    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py>`_.

    Args:
        predictor (dict): Config dict for module of predictor.
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 1.0.
    """

    def __init__(self,
                 predictor: dict,
                 loss: dict,
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.predictor = MODELS.build(predictor)
        self.loss_module = MODELS.build(loss)
        self.temperature = temperature

    def loss(self, base_out: torch.Tensor,
             momentum_out: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            base_out (torch.Tensor): NxC features from base_encoder.
            momentum_out (torch.Tensor): NxC features from momentum_encoder.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor([base_out])[0]

        # normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(momentum_out, dim=1)

        # get negative samples
        target = torch.cat(all_gather(target), dim=0)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [pred, target]) / self.temperature

        # generate labels
        batch_size = logits.shape[0]
        labels = (torch.arange(batch_size, dtype=torch.long) +
                  batch_size * get_rank()).to(logits.device)

        loss = self.loss_module(logits, labels)
        return loss
