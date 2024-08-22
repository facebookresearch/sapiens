# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class ToyModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, *args, **kwargs):
        return {'loss': torch.tensor(0.0)}


def update_params_step(self, loss):
    pass


def runtimeinfo_step(self, runner, batch_idx, data_batch=None):
    runner.message_hub.update_info('iter', runner.iter)
    lr_dict = runner.optim_wrapper.get_lr()
    for name, lr in lr_dict.items():
        runner.message_hub.update_scalar(f'train/{name}', lr[0])

    momentum_dict = runner.optim_wrapper.get_momentum()
    for name, momentum in momentum_dict.items():
        runner.message_hub.update_scalar(f'train/{name}', momentum[0])


@patch('mmengine.optim.optimizer.OptimWrapper.update_params',
       update_params_step)
@patch('mmengine.hooks.RuntimeInfoHook.before_train_iter', runtimeinfo_step)
def fake_run(cfg):
    from mmengine.runner import Runner
    cfg.pop('model')
    cfg.pop('visualizer')
    cfg.pop('val_dataloader')
    cfg.pop('val_evaluator')
    cfg.pop('val_cfg')
    cfg.pop('test_dataloader')
    cfg.pop('test_evaluator')
    cfg.pop('test_cfg')
    extra_cfg = dict(
        model=dict(type='ToyModel'),
        visualizer=dict(
            type='Visualizer',
            vis_backends=[
                dict(type='TensorboardVisBackend', save_dir='temp_dir')
            ]),
    )
    cfg.merge_from_dict(extra_cfg)
    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
