# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmpretrain.utils.dependency import WITH_MULTIMODAL

from .base_dataset import BaseDataset
from .builder import build_dataset
from .caltech101 import Caltech101
from .cifar import CIFAR10, CIFAR100
from .cub import CUB
from .custom import CustomDataset
from .dataset_wrappers import KFoldDataset
from .dtd import DTD
from .fgvcaircraft import FGVCAircraft
from .flowers102 import Flowers102
from .food101 import Food101
from .imagenet import ImageNet, ImageNet21k
from .inshop import InShop
from .mnist import FashionMNIST, MNIST
from .multi_label import MultiLabelDataset
from .multi_task import MultiTaskDataset
from .nlvr2 import NLVR2
from .oxfordiiitpet import OxfordIIITPet
from .places205 import Places205
from .samplers import *  # noqa: F401,F403
from .stanfordcars import StanfordCars
from .sun397 import SUN397
from .transforms import *  # noqa: F401,F403
from .voc import VOC

__all__ = [
    "BaseDataset",
    "CIFAR10",
    "CIFAR100",
    "CUB",
    "Caltech101",
    "CustomDataset",
    "DTD",
    "FGVCAircraft",
    "FashionMNIST",
    "Flowers102",
    "Food101",
    "ImageNet",
    "ImageNet21k",
    "InShop",
    "KFoldDataset",
    "MNIST",
    "MultiLabelDataset",
    "MultiTaskDataset",
    "NLVR2",
    "OxfordIIITPet",
    "Places205",
    "SUN397",
    "StanfordCars",
    "VOC",
    "build_dataset",
]

try:
    from .dataset_wrappers import CombinedDataset
    from .humans_300m import Humans300M

    __all__ += ["Humans300M", "CombinedDataset"]

except Exception as e:
    pass

if WITH_MULTIMODAL:
    from .coco_caption import COCOCaption
    from .coco_retrieval import COCORetrieval
    from .coco_vqa import COCOVQA
    from .flamingo import FlamingoEvalCOCOCaption, FlamingoEvalCOCOVQA
    from .flickr30k_caption import Flickr30kCaption
    from .flickr30k_retrieval import Flickr30kRetrieval
    from .gqa_dataset import GQA
    from .iconqa import IconQA
    from .infographic_vqa import InfographicVQA
    from .nocaps import NoCaps
    from .ocr_vqa import OCRVQA
    from .refcoco import RefCOCO
    from .scienceqa import ScienceQA
    from .textvqa import TextVQA
    from .visual_genome import VisualGenomeQA
    from .vizwiz import VizWiz
    from .vsr import VSR

    __all__.extend(
        [
            "COCOCaption",
            "COCORetrieval",
            "COCOVQA",
            "FlamingoEvalCOCOCaption",
            "FlamingoEvalCOCOVQA",
            "Flickr30kCaption",
            "Flickr30kRetrieval",
            "RefCOCO",
            "VisualGenomeQA",
            "ScienceQA",
            "NoCaps",
            "GQA",
            "TextVQA",
            "VSR",
            "VizWiz",
            "OCRVQA",
            "InfographicVQA",
            "IconQA",
        ]
    )
