# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .alexnet import AlexNet
from .beit import BEiTViT
from .conformer import Conformer
from .convmixer import ConvMixer
from .convnext import ConvNeXt
from .cspnet import CSPDarkNet, CSPNet, CSPResNet, CSPResNeXt
from .davit import DaViT
from .deit import DistilledVisionTransformer
from .deit3 import DeiT3
from .densenet import DenseNet
from .edgenext import EdgeNeXt
from .efficientformer import EfficientFormer
from .efficientnet import EfficientNet
from .efficientnet_v2 import EfficientNetV2
from .hivit import HiViT
from .hornet import HorNet
from .hrnet import HRNet
from .inception_v3 import InceptionV3
from .lenet import LeNet5
from .levit import LeViT
from .mixmim import MixMIMTransformer
from .mlp_mixer import MlpMixer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mobileone import MobileOne
from .mobilevit import MobileViT
from .mvit import MViT
from .poolformer import PoolFormer
from .regnet import RegNet
from .replknet import RepLKNet
from .repmlp import RepMLPNet
from .repvgg import RepVGG
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .resnext import ResNeXt
from .revvit import RevVisionTransformer
from .riformer import RIFormer
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .sparse_convnext import SparseConvNeXt
from .sparse_resnet import SparseResNet
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .t2t_vit import T2T_ViT
from .timm_backbone import TIMMBackbone
from .tinyvit import TinyViT
from .tnt import TNT
from .twins import PCPVT, SVT
from .van import VAN
from .vgg import VGG
from .vig import PyramidVig, Vig
from .vision_transformer import VisionTransformer, TransformerEncoderLayer
from .vision_transformer2 import VisionTransformer2
from .vit_eva02 import ViTEVA02
from .vit_sam import ViTSAM
from .xcit import XCiT

__all__ = [
    'LeNet5',
    'AlexNet',
    'VGG',
    'RegNet',
    'ResNet',
    'ResNeXt',
    'ResNetV1d',
    'ResNeSt',
    'ResNet_CIFAR',
    'SEResNet',
    'SEResNeXt',
    'ShuffleNetV1',
    'ShuffleNetV2',
    'MobileNetV2',
    'MobileNetV3',
    'VisionTransformer',
    'TransformerEncoderLayer',
    'SwinTransformer',
    'TNT',
    'TIMMBackbone',
    'T2T_ViT',
    'Res2Net',
    'RepVGG',
    'Conformer',
    'MlpMixer',
    'DistilledVisionTransformer',
    'PCPVT',
    'SVT',
    'EfficientNet',
    'EfficientNetV2',
    'ConvNeXt',
    'HRNet',
    'ResNetV1c',
    'ConvMixer',
    'EdgeNeXt',
    'CSPDarkNet',
    'CSPResNet',
    'CSPResNeXt',
    'CSPNet',
    'RepLKNet',
    'RepMLPNet',
    'PoolFormer',
    'RIFormer',
    'DenseNet',
    'VAN',
    'InceptionV3',
    'MobileOne',
    'EfficientFormer',
    'SwinTransformerV2',
    'MViT',
    'DeiT3',
    'HorNet',
    'MobileViT',
    'DaViT',
    'BEiTViT',
    'RevVisionTransformer',
    'MixMIMTransformer',
    'TinyViT',
    'LeViT',
    'Vig',
    'PyramidVig',
    'XCiT',
    'ViTSAM',
    'ViTEVA02',
    'HiViT',
    'SparseResNet',
    'SparseConvNeXt',
    'VisionTransformer2',
]
