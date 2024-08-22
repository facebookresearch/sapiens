# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .h36m_dataset import Human36mDataset
from .goliath3d_dataset import Goliath3dDataset
from .aic2goliath3d_dataset import Aic2Goliath3dDataset
from .coco_wholebody2goliath3d_dataset import CocoWholeBody2Goliath3dDataset
from .crowdpose2goliath3d_dataset import Crowdpose2Goliath3dDataset
from .mpii2goliath3d_dataset import Mpii2Goliath3dDataset

__all__ = ['Human36mDataset', 'Goliath3dDataset', 'Aic2Goliath3dDataset', 'CocoWholeBody2Goliath3dDataset', 'Crowdpose2Goliath3dDataset', 'Mpii2Goliath3dDataset']
