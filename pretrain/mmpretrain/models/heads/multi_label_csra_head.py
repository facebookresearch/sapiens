# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/Kevinz-code/CSRA
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from .multi_label_cls_head import MultiLabelClsHead


@MODELS.register_module()
class CSRAClsHead(MultiLabelClsHead):
    """Class-specific residual attention classifier head.

    Please refer to the `Residual Attention: A Simple but Effective Method for
    Multi-Label Recognition (ICCV 2021) <https://arxiv.org/abs/2108.02456>`_
    for details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to use ``dict(type='Normal', layer='Linear', std=0.01)``.
    """
    temperature_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_heads: int,
                 lam: float,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        assert num_heads in self.temperature_settings.keys(
        ), 'The num of heads is not in temperature setting.'
        assert lam > 0, 'Lambda should be between 0 and 1.'
        super(CSRAClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
        self.temp_list = self.temperature_settings[num_heads]
        self.csra_heads = ModuleList([
            CSRAModule(num_classes, in_channels, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``CSRAClsHead``, we just obtain the
        feature of the last stage.
        """
        # The CSRAClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        logit = sum([head(pre_logits) for head in self.csra_heads])
        return logit


class CSRAModule(BaseModule):
    """Basic module of CSRA with different temperature.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        T (int): Temperature setting.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 T: int,
                 lam: float,
                 init_cfg=None):

        super(CSRAModule, self).__init__(init_cfg=init_cfg)
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(in_channels, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        score = self.head(x) / torch.norm(
            self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)

        return base_logit + self.lam * att_logit
