# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from .averaged_model import (BaseAveragedModel, ExponentialMovingAverage,
                             MomentumAnnealingEMA, StochasticWeightAverage)
from .base_model import BaseDataPreprocessor, BaseModel, ImgDataPreprocessor
from .base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .test_time_aug import BaseTTAModel
from .utils import (convert_sync_batchnorm, detect_anomalous_params,
                    merge_dict, revert_sync_batchnorm, stack_batch)
from .weight_init import (BaseInit, Caffe2XavierInit, ConstantInit,
                          KaimingInit, NormalInit, PretrainedInit,
                          TruncNormalInit, UniformInit, XavierInit,
                          bias_init_with_prob, caffe2_xavier_init,
                          constant_init, initialize, kaiming_init, normal_init,
                          trunc_normal_init, uniform_init, update_init_info,
                          xavier_init)
from .wrappers import (MMDistributedDataParallel,
                       MMSeparateDistributedDataParallel, is_model_wrapper)

__all__ = [
    'MMDistributedDataParallel', 'is_model_wrapper', 'BaseAveragedModel',
    'StochasticWeightAverage', 'ExponentialMovingAverage',
    'MomentumAnnealingEMA', 'BaseModel', 'BaseDataPreprocessor',
    'ImgDataPreprocessor', 'MMSeparateDistributedDataParallel', 'BaseModule',
    'stack_batch', 'merge_dict', 'detect_anomalous_params', 'ModuleList',
    'ModuleDict', 'Sequential', 'revert_sync_batchnorm', 'update_init_info',
    'constant_init', 'xavier_init', 'normal_init', 'trunc_normal_init',
    'uniform_init', 'kaiming_init', 'caffe2_xavier_init',
    'bias_init_with_prob', 'BaseInit', 'ConstantInit', 'XavierInit',
    'NormalInit', 'TruncNormalInit', 'UniformInit', 'KaimingInit',
    'Caffe2XavierInit', 'PretrainedInit', 'initialize',
    'convert_sync_batchnorm', 'BaseTTAModel'
]

if digit_version(TORCH_VERSION) >= digit_version('2.0.0'):
    from .wrappers import MMFullyShardedDataParallel  # noqa:F401
    __all__.append('MMFullyShardedDataParallel')
