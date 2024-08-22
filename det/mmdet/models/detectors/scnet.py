# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmdet.registry import MODELS
from .cascade_rcnn import CascadeRCNN


@MODELS.register_module()
class SCNet(CascadeRCNN):
    """Implementation of `SCNet <https://arxiv.org/abs/2012.10150>`_"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
