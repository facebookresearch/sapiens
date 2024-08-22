# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmdet.registry import MODELS
from .cascade_rcnn import CascadeRCNN


@MODELS.register_module()
class HybridTaskCascade(CascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def with_semantic(self) -> bool:
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
