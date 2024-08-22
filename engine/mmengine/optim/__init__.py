# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .optimizer import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS,
                        AmpOptimWrapper, ApexOptimWrapper, BaseOptimWrapper,
                        DefaultOptimWrapperConstructor, OptimWrapper,
                        OptimWrapperDict, ZeroRedundancyOptimizer,
                        build_optim_wrapper)
# yapf: disable
from .scheduler import (ConstantLR, ConstantMomentum, ConstantParamScheduler,
                        CosineAnnealingLR, CosineAnnealingMomentum,
                        CosineAnnealingParamScheduler, ExponentialLR,
                        ExponentialMomentum, ExponentialParamScheduler,
                        LinearLR, LinearMomentum, LinearParamScheduler,
                        MultiStepLR, MultiStepMomentum,
                        MultiStepParamScheduler, OneCycleLR,
                        OneCycleParamScheduler, PolyLR, PolyMomentum,
                        PolyParamScheduler, ReduceOnPlateauLR,
                        ReduceOnPlateauMomentum, ReduceOnPlateauParamScheduler,
                        StepLR, StepMomentum, StepParamScheduler,
                        _ParamScheduler)

# yapf: enable
__all__ = [
    'OPTIM_WRAPPER_CONSTRUCTORS', 'OPTIMIZERS', 'build_optim_wrapper',
    'DefaultOptimWrapperConstructor', 'ConstantLR', 'CosineAnnealingLR',
    'ExponentialLR', 'LinearLR', 'MultiStepLR', 'StepLR', 'ConstantMomentum',
    'CosineAnnealingMomentum', 'ExponentialMomentum', 'LinearMomentum',
    'MultiStepMomentum', 'StepMomentum', 'ConstantParamScheduler',
    'CosineAnnealingParamScheduler', 'ExponentialParamScheduler',
    'LinearParamScheduler', 'MultiStepParamScheduler', 'StepParamScheduler',
    '_ParamScheduler', 'OptimWrapper', 'AmpOptimWrapper', 'ApexOptimWrapper',
    'OptimWrapperDict', 'OneCycleParamScheduler', 'OneCycleLR', 'PolyLR',
    'PolyMomentum', 'PolyParamScheduler', 'ReduceOnPlateauLR',
    'ReduceOnPlateauMomentum', 'ReduceOnPlateauParamScheduler',
    'ZeroRedundancyOptimizer', 'BaseOptimWrapper'
]
