# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.models.heads import ClsHead
from mmpretrain.registry import MODELS
from ..utils import build_norm_layer


class BatchNormLinear(BaseModule):

    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN1d')):
        super(BatchNormLinear, self).__init__()
        self.bn = build_norm_layer(norm_cfg, in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    @torch.no_grad()
    def fuse(self):
        w = self.bn.weight / (self.bn.running_var + self.bn.eps)**0.5
        b = self.bn.bias - self.bn.running_mean * \
            self.bn.weight / (self.bn.running_var + self.bn.eps) ** 0.5
        w = self.linear.weight * w[None, :]
        b = (self.linear.weight @ b[:, None]).view(-1) + self.linear.bias

        self.linear.weight.data.copy_(w)
        self.linear.bias.data.copy_(b)
        return self.linear

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x


def fuse_parameters(module):
    for child_name, child in module.named_children():
        if hasattr(child, 'fuse'):
            setattr(module, child_name, child.fuse())
        else:
            fuse_parameters(child)


@MODELS.register_module()
class LeViTClsHead(ClsHead):

    def __init__(self,
                 num_classes=1000,
                 distillation=True,
                 in_channels=None,
                 deploy=False,
                 **kwargs):
        super(LeViTClsHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.distillation = distillation
        self.deploy = deploy
        self.head = BatchNormLinear(in_channels, num_classes)
        if distillation:
            self.head_dist = BatchNormLinear(in_channels, num_classes)

        if self.deploy:
            self.switch_to_deploy(self)

    def switch_to_deploy(self):
        if self.deploy:
            return
        fuse_parameters(self)
        self.deploy = True

    def forward(self, x):
        x = self.pre_logits(x)
        if self.distillation:
            x = self.head(x), self.head_dist(x)  # 2 16 384 -> 2 1000
            if not self.training:
                x = (x[0] + x[1]) / 2
            else:
                raise NotImplementedError("MMPretrain doesn't support "
                                          'training in distillation mode.')
        else:
            x = self.head(x)
        return x
