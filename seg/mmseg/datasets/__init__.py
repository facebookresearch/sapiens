# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# yapf: disable
from .ade import ADE20KDataset
from .basesegdataset import BaseCDDataset, BaseSegDataset
from .bdd100k import BDD100KDataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .dataset_wrappers import MultiImageMixDataset
from .lip import LIPDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59

from .goliath import GoliathDataset
from .lip2goliath import LIP2GoliathDataset
from .cihp2goliath import CIHP2GoliathDataset
from .atr2goliath import ATR2GoliathDataset
from .pascal2goliath import Pascal2GoliathDataset
from .face import FaceDataset

from .render_people import RenderPeopleDataset
from .metric_render_people import MetricRenderPeopleDataset
from .depth_dataset_wrappers import DepthCombinedDataset
from .normal_render_people import NormalRenderPeopleDataset
from .normal_dataset_wrappers import NormalCombinedDataset
from .albedo import AlbedoDataset
from .hdri import HDRIDataset
from .normal_general import NormalGeneralDataset
from .depth_general import DepthGeneralDataset
from .albedo_render_people import AlbedoRenderPeopleDataset
from .pointmap_render_people import PointmapRenderPeopleDataset
from .pointmap_dataset_wrappers import PointmapCombinedDataset
from .stereo_pointmap_render_people import StereoPointmapRenderPeopleDataset
from .stereo_pointmap_dataset_wrappers import StereoPointmapCombinedDataset
from .general_dataset_wrappers import GeneralCombinedDataset
from .stereo_correspondences_render_people import StereoCorrespondencesRenderPeopleDataset
from .stereo_correspondences_dataset_wrappers import StereoCorrespondencesCombinedDataset

from .cihp import CIHPDataset
from .voc import PascalVOCDataset

# yapf: disable
from .transforms import (CLAHE, AdjustGamma, Albu, BioMedical3DPad,
                         BioMedical3DRandomCrop, BioMedical3DRandomFlip,
                         BioMedicalGaussianBlur, BioMedicalGaussianNoise,
                         BioMedicalRandomGamma, ConcatCDInput, GenerateEdge,
                         LoadAnnotations, LoadBiomedicalAnnotation,
                         LoadBiomedicalData, LoadBiomedicalImageFromFile,
                         LoadImageFromNDArray, LoadMultipleRSImageFromFile,
                         LoadSingleRSImageFromFile, PackSegInputs,
                         PhotoMetricDistortion, RandomCrop, RandomCutOut,
                         RandomMosaic, RandomRotate, RandomRotFlip, Rerange,
                         ResizeShortestEdge, ResizeToMultiple, RGB2Gray,
                         LoadImage,
                         SegRescale)

from .transforms.depth_transforms import RandomDepthResizeCompensate, DepthRandomFlip, \
                    RandomDepthCrop, DepthResize, DepthRandomRotate, GenerateDepthTarget, GenerateMetricDepthTarget, PackDepthInputs

from .transforms.normal_transforms import RandomNormalResizeCompensate, NormalRandomFlip, \
                    RandomNormalCrop, NormalResize, GenerateNormalTarget, PackNormalInputs

from .transforms.albedo_transforms import RandomAlbedoResizeCompensate, AlbedoRandomFlip, \
                    RandomAlbedoCrop, AlbedoResize, AlbedoRandomRotate, GenerateAlbedoTarget, PackAlbedoInputs

from .transforms.hdri_transforms import HDRIResize, GenerateHDRITarget, PackHDRIInputs

from .transforms.pointmap_transforms import RandomPointmapResizeCompensate, PointmapRandomFlip, \
                    RandomPointmapCrop, PointmapResize, GeneratePointmapTarget, PackPointmapInputs, PadPointmap \

from .transforms.stereo_pointmap_transforms import PackStereoPointmapInputs, TestPackStereoPointmapInputs
from .transforms.stereo_correspondences_transforms import PackStereoCorrespondencesInputs

# yapf: enable
__all__ = [
    'BaseSegDataset', 'BioMedical3DRandomCrop', 'BioMedical3DRandomFlip',
    'CityscapesDataset', 'PascalVOCDataset', 'ADE20KDataset',
    'PascalContextDataset', 'PascalContextDataset59', 'ChaseDB1Dataset',
    'DRIVEDataset', 'HRFDataset', 'STAREDataset', 'DarkZurichDataset',
    'NightDrivingDataset', 'COCOStuffDataset', 'LoveDADataset',
    'MultiImageMixDataset', 'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset',
    'LoadAnnotations', 'RandomCrop', 'SegRescale', 'PhotoMetricDistortion',
    'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange', 'RGB2Gray',
    'RandomCutOut', 'RandomMosaic', 'PackSegInputs', 'ResizeToMultiple',
    'LoadImageFromNDArray', 'LoadBiomedicalImageFromFile',
    'LoadBiomedicalAnnotation', 'LoadBiomedicalData', 'GenerateEdge',
    'DecathlonDataset', 'LIPDataset', 'ResizeShortestEdge',
    'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur',
    'BioMedicalRandomGamma', 'BioMedical3DPad', 'RandomRotFlip',
    'SynapseDataset', 'REFUGEDataset', 'MapillaryDataset_v1',
    'MapillaryDataset_v2', 'Albu', 'LEVIRCDDataset',
    'LoadMultipleRSImageFromFile', 'LoadSingleRSImageFromFile',
    'ConcatCDInput', 'BaseCDDataset', 'DSDLSegDataset', 'BDD100KDataset',
    'NYUDataset', 'LoadImage',
    'GoliathDataset', 'LIP2GoliathDataset', 'CIHP2GoliathDataset',
    'ATR2GoliathDataset', 'Pascal2GoliathDataset', 'FaceDataset',
    'DepthCombinedDataset', 'CIHPDataset', 'NormalGeneralDataset',
    'GeneralCombinedDataset', 
]
