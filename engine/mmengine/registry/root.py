# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MMEngine provides 20 root registries to support using modules across
projects.

More datails can be found at
https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html.
"""

from .build_functions import (build_model_from_cfg, build_runner_from_cfg,
                              build_scheduler_from_cfg)
from .registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', build_func=build_runner_from_cfg)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry('runner constructor')
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop')
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook')

# manage all kinds of strategies like `NativeStrategy` and `DDPStrategy`
STRATEGIES = Registry('strategy')

# manage data-related modules
DATASETS = Registry('dataset')
DATA_SAMPLERS = Registry('data sampler')
TRANSFORMS = Registry('transform')

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', build_model_from_cfg)
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper')
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry('weight initializer')

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer')
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper')
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry('optimizer wrapper constructor')
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', build_func=build_scheduler_from_cfg)

# manage all kinds of metrics
METRICS = Registry('metric')
# manage evaluator
EVALUATOR = Registry('evaluator')

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util')

# manage visualizer
VISUALIZERS = Registry('visualizer')
# manage visualizer backend
VISBACKENDS = Registry('vis_backend')

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor')

# manage inferencer
INFERENCERS = Registry('inferencer')

# manage function
FUNCTIONS = Registry('function')
