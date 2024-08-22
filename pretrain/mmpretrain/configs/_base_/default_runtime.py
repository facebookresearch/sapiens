# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This is a BETA new format config file, and the usage may change recently.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.visualization import LocalVisBackend

from mmpretrain.engine.hooks import VisualizationHook
from mmpretrain.visualization import UniversalVisualizer

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),

    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),

    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),

    # validation results visualization, set True to enable it.
    visualization=dict(type=VisualizationHook, enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(type=UniversalVisualizer, vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# Do not need to specify default_scope with new config. Therefore set it to
# None to avoid BC-breaking.
default_scope = None
