# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.model import ImgDataPreprocessor

from mmpose.registry import MODELS


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""
