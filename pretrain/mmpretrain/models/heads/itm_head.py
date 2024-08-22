# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.evaluation import Accuracy
from mmpretrain.registry import MODELS


class Pooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@MODELS.register_module()
class ITMHead(BaseModule):
    """Image-text matching head for multi-modal pre-trained task. Adapted by
    BLIP, FLAVA.

    Args:
        hidden_size (int): Hidden channel size out input features.
        with_pooler (bool): Whether a pooler is added. Defaults to True.
        loss (dict): Config of global contrasive loss. Defaults to
            ``dict(type='GlobalContrasiveLoss')``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 hidden_size: int,
                 with_pooler: bool = True,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 cal_acc: bool = False,
                 init_cfg: Optional[dict] = None):
        super(ITMHead, self).__init__(init_cfg=init_cfg)
        self.hidden_size = hidden_size

        if with_pooler:
            self.pooler = Pooler(hidden_size=self.hidden_size)
        else:
            self.pooler = nn.Identity()
        self.fc = nn.Linear(self.hidden_size, 2)

        self.loss_module = MODELS.build(loss)
        self.cal_acc = cal_acc

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pooler(feats[-1])
        itm_logits = self.fc(pre_logits)
        return itm_logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples, **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # The part can be traced by torch.fx
        itm_logits = self(feats)

        # deal with query
        if itm_logits.ndim == 3:
            itm_logits = itm_logits.mean(dim=1)

        # The part can not be traced by torch.fx
        losses = self._get_loss(itm_logits, data_samples, **kwargs)
        return losses

    def _get_loss(self, itm_logits: torch.Tensor, data_samples, **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        # use `itm_label` in here temporarily
        target = torch.tensor([i.is_matched
                               for i in data_samples]).to(itm_logits.device)

        # compute loss
        losses = dict()

        loss = self.loss_module(
            itm_logits, target.long(), avg_factor=itm_logits.size(0), **kwargs)
        losses['itm_loss'] = loss

        # compute accuracy
        if self.cal_acc:
            # topk is meaningless for matching task
            acc = Accuracy.calculate(itm_logits, target)
            # acc is warpped with two lists of topk and thrs
            # which are unnecessary here
            losses.update({'itm_accuracy': acc[0][0]})

        return losses
