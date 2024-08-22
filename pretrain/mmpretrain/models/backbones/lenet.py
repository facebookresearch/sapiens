# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


@MODELS.register_module()
class LeNet5(BaseBackbone):
    """`LeNet5 <https://en.wikipedia.org/wiki/LeNet>`_ backbone.

    The input for LeNet-5 is a 32×32 grayscale image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=-1):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1), nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1), nn.Tanh())
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes),
            )

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x.squeeze())

        return (x, )
