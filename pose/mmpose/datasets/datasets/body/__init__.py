# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .aic_dataset import AicDataset
from .coco_dataset import CocoDataset
from .crowdpose_dataset import CrowdPoseDataset
from .humanart_dataset import HumanArtDataset
from .jhmdb_dataset import JhmdbDataset
from .mhp_dataset import MhpDataset
from .mpii_dataset import MpiiDataset
from .mpii_trb_dataset import MpiiTrbDataset
from .ochuman_dataset import OCHumanDataset
from .posetrack18_dataset import PoseTrack18Dataset
from .posetrack18_video_dataset import PoseTrack18VideoDataset
from .aic2coco_dataset import Aic2CocoDataset
from .mpii2coco_dataset import Mpii2CocoDataset
from .crowdpose2coco_dataset import Crowdpose2CocoDataset
from .goliath_dataset import GoliathDataset
from .goliath_eval_dataset import GoliathEvalDataset
from .goliath3d_eval_dataset import Goliath3dEvalDataset
from .coco2goliath_dataset import Coco2GoliathDataset
from .crowdpose2goliath_dataset import Crowdpose2GoliathDataset
from .aic2goliath_dataset import Aic2GoliathDataset
from .mpii2goliath_dataset import Mpii2GoliathDataset

__all__ = [
    'CocoDataset', 'MpiiDataset', 'MpiiTrbDataset', 'AicDataset',
    'CrowdPoseDataset', 'OCHumanDataset', 'MhpDataset', 'PoseTrack18Dataset',
    'JhmdbDataset', 'PoseTrack18VideoDataset', 'HumanArtDataset', 'Aic2CocoDataset',
    'Mpii2CocoDataset', 'Crowdpose2CocoDataset', 'GoliathDataset', 'GoliathEvalDataset',
    'Coco2GoliathDataset', 'Crowdpose2GoliathDataset', 'Mpii2GoliathDataset', 'Aic2GoliathDataset',
    'Goliath3dEvalDataset'
]
