# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .ddr_head import DDRHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .ham_head import LightHamHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .pid_head import PIDHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .san_head import SideAdapterCLIPHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .vpd_depth_head import VPDDepthHead
from .vit_head import VitHead
from .vit_depth_head import VitDepthHead
from .vit_metric_depth_head import VitMetricDepthHead
from .vit_normal_head import VitNormalHead
from .vit_hdri_head import VitHDRIHead
from .vit_pointmap_head import VitPointmapHead
from .vit_stereo_pointmap_head import VitStereoPointmapHead
from .vit_stereo_correspondences_head import VitStereoCorrespondencesHead
from .vit_normal_head2 import VitNormalHead2

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'STDCHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator', 'MaskFormerHead', 'Mask2FormerHead',
    'LightHamHead', 'PIDHead', 'DDRHead', 'VPDDepthHead', 'SideAdapterCLIPHead',
    'VitHead', 'VitDepthHead', 'VitNormalHead', 'VitHDRIHead', 'VitPointmapHead',
    'VitStereoPointmapHead', 'VitMetricDepthHead', 'VitStereoCorrespondencesHead',
    'VitNormalHead2'
]
