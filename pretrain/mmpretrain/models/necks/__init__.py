# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .beitv2_neck import BEiTV2Neck
from .cae_neck import CAENeck
from .densecl_neck import DenseCLNeck
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .itpn_neck import iTPNPretrainDecoder
from .linear_neck import LinearNeck
from .mae_neck import ClsBatchNormNeck, MAEPretrainDecoder
from .milan_neck import MILANPretrainDecoder
from .mixmim_neck import MixMIMPretrainDecoder
from .mocov2_neck import MoCoV2Neck
from .nonlinear_neck import NonLinearNeck
from .simmim_neck import SimMIMLinearDecoder
from .spark_neck import SparKLightDecoder
from .swav_neck import SwAVNeck
from .mae_neck2 import MAEPretrainDecoder2

__all__ = [
    'GlobalAveragePooling',
    'GeneralizedMeanPooling',
    'HRFuseScales',
    'LinearNeck',
    'BEiTV2Neck',
    'CAENeck',
    'DenseCLNeck',
    'MAEPretrainDecoder',
    'ClsBatchNormNeck',
    'MILANPretrainDecoder',
    'MixMIMPretrainDecoder',
    'MoCoV2Neck',
    'NonLinearNeck',
    'SimMIMLinearDecoder',
    'SwAVNeck',
    'iTPNPretrainDecoder',
    'SparKLightDecoder',
    'MAEPretrainDecoder2',
]
