# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .class_num_check_hook import ClassNumCheckHook
from .densecl_hook import DenseCLHook
from .ema_hook import EMAHook
from .margin_head_hooks import SetAdaptiveMarginsHook
from .precise_bn_hook import PreciseBNHook
from .retriever_hooks import PrepareProtoBeforeValLoopHook
from .simsiam_hook import SimSiamHook
from .swav_hook import SwAVHook
from .switch_recipe_hook import SwitchRecipeHook
from .visualization_hook import VisualizationHook
from .warmup_param_hook import WarmupParamHook
from .pretrain_visualization_hook import PretrainVisualizationHook
from .pretrain2_visualization_hook import Pretrain2VisualizationHook

__all__ = [
    'ClassNumCheckHook', 'PreciseBNHook', 'VisualizationHook',
    'SwitchRecipeHook', 'PrepareProtoBeforeValLoopHook',
    'SetAdaptiveMarginsHook', 'EMAHook', 'SimSiamHook', 'DenseCLHook',
    'SwAVHook', 'WarmupParamHook', 'PretrainVisualizationHook', 'Pretrain2VisualizationHook',
]
