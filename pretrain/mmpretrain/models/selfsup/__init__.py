# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .barlowtwins import BarlowTwins
from .base import BaseSelfSupervisor
from .beit import VQKD, BEiT, BEiTPretrainViT
from .byol import BYOL
from .cae import CAE, CAEPretrainViT, DALLEEncoder
from .densecl import DenseCL
from .eva import EVA
from .itpn import iTPN, iTPNHiViT
from .mae import MAE, MAEHiViT, MAEViT
from .maskfeat import HOGGenerator, MaskFeat, MaskFeatViT
from .mff import MFF, MFFViT
from .milan import MILAN, CLIPGenerator, MILANViT
from .mixmim import MixMIM, MixMIMPretrainTransformer
from .moco import MoCo
from .mocov3 import MoCoV3, MoCoV3ViT
from .simclr import SimCLR
from .simmim import SimMIM, SimMIMSwinTransformer
from .simsiam import SimSiam
from .spark import SparK
from .swav import SwAV
from .mae_eva02 import MAEViTEVA02
from .mae_sapiens2 import MAEViT2

__all__ = [
    'BaseSelfSupervisor',
    'BEiTPretrainViT',
    'VQKD',
    'CAEPretrainViT',
    'DALLEEncoder',
    'MAEViT',
    'MAEHiViT',
    'iTPNHiViT',
    'iTPN',
    'HOGGenerator',
    'MaskFeatViT',
    'CLIPGenerator',
    'MILANViT',
    'MixMIMPretrainTransformer',
    'MoCoV3ViT',
    'SimMIMSwinTransformer',
    'MoCo',
    'MoCoV3',
    'BYOL',
    'SimCLR',
    'SimSiam',
    'BEiT',
    'CAE',
    'MAE',
    'MaskFeat',
    'MILAN',
    'MixMIM',
    'SimMIM',
    'EVA',
    'DenseCL',
    'BarlowTwins',
    'SwAV',
    'SparK',
    'MFF',
    'MFFViT',
    'MAEViTEVA02',
    'MAEViT2',
]
