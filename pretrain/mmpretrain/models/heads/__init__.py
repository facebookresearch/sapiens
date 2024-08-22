# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .beitv1_head import BEiTV1Head
from .beitv2_head import BEiTV2Head
from .cae_head import CAEHead
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .contrastive_head import ContrastiveHead
from .deit_head import DeiTClsHead
from .efficientformer_head import EfficientFormerClsHead
from .grounding_head import GroundingHead
from .itc_head import ITCHead
from .itm_head import ITMHead
from .itpn_clip_head import iTPNClipHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .levit_head import LeViTClsHead
from .linear_head import LinearClsHead
from .mae_head import MAEPretrainHead
from .mae_head2 import MAEPretrainHead2
from .margin_head import ArcFaceClsHead
from .mim_head import MIMHead
from .mixmim_head import MixMIMPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_label_cls_head import MultiLabelClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multi_task_head import MultiTaskHead
from .seq_gen_head import SeqGenerationHead
from .simmim_head import SimMIMHead
from .spark_head import SparKPretrainHead
from .stacked_head import StackedLinearClsHead
from .swav_head import SwAVHead
from .vig_head import VigClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .vqa_head import VQAGenerationHead

__all__ = [
    'ClsHead',
    'LinearClsHead',
    'StackedLinearClsHead',
    'MultiLabelClsHead',
    'MultiLabelLinearClsHead',
    'VisionTransformerClsHead',
    'DeiTClsHead',
    'ConformerHead',
    'EfficientFormerClsHead',
    'ArcFaceClsHead',
    'CSRAClsHead',
    'MultiTaskHead',
    'LeViTClsHead',
    'VigClsHead',
    'BEiTV1Head',
    'BEiTV2Head',
    'CAEHead',
    'ContrastiveHead',
    'LatentCrossCorrelationHead',
    'LatentPredictHead',
    'MAEPretrainHead',
    'MixMIMPretrainHead',
    'SwAVHead',
    'MoCoV3Head',
    'MIMHead',
    'SimMIMHead',
    'SeqGenerationHead',
    'VQAGenerationHead',
    'ITCHead',
    'ITMHead',
    'GroundingHead',
    'iTPNClipHead',
    'SparKPretrainHead',
]
