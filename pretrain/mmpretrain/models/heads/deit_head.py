# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .vision_transformer_head import VisionTransformerClsHead


@MODELS.register_module()
class DeiTClsHead(VisionTransformerClsHead):
    """Distilled Vision Transformer classifier head.

    Comparing with the :class:`VisionTransformerClsHead`, this head adds an
    extra linear layer to handle the dist token. The final classification score
    is the average of both linear transformation results of ``cls_token`` and
    ``dist_token``.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (int, optional): Number of the dimensions for hidden layer.
            Defaults to None, which means no extra hidden layer.
        act_cfg (dict): The activation config. Only available during
            pre-training. Defaults to ``dict(type='Tanh')``.
        init_cfg (dict): The extra initialization configs. Defaults to
            ``dict(type='Constant', layer='Linear', val=0)``.
    """

    def _init_layers(self):
        """"Init extra hidden linear layer to handle dist token if exists."""
        super(DeiTClsHead, self)._init_layers()
        if self.hidden_dim is None:
            head_dist = nn.Linear(self.in_channels, self.num_classes)
        else:
            head_dist = nn.Linear(self.hidden_dim, self.num_classes)
        self.layers.add_module('head_dist', head_dist)

    def pre_logits(self,
                   feats: Tuple[List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage. In ``DeiTClsHead``, we obtain the
        feature of the last stage and forward in hidden layer if exists.
        """
        feat = feats[-1]  # Obtain feature of the last scale.
        # For backward-compatibility with the previous ViT output
        if len(feat) == 3:
            _, cls_token, dist_token = feat
        else:
            cls_token, dist_token = feat
        if self.hidden_dim is None:
            return cls_token, dist_token
        else:
            cls_token = self.layers.act(self.layers.pre_logits(cls_token))
            dist_token = self.layers.act(self.layers.pre_logits(dist_token))
            return cls_token, dist_token

    def forward(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The forward process."""
        if self.training:
            warnings.warn('MMPretrain cannot train the '
                          'distilled version DeiT.')
        cls_token, dist_token = self.pre_logits(feats)
        # The final classification head.
        cls_score = (self.layers.head(cls_token) +
                     self.layers.head_dist(dist_token)) / 2
        return cls_score
