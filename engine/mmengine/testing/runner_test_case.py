# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
import shutil
import tempfile
import time
from unittest import TestCase
from uuid import uuid4

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group
from torch.utils.data import Dataset

import mmengine.hooks  # noqa F401
import mmengine.optim  # noqa F401
from mmengine.config import Config
from mmengine.dist import is_distributed
from mmengine.evaluator import BaseMetric
from mmengine.logging import MessageHub, MMLogger
from mmengine.model import BaseModel
from mmengine.registry import DATASETS, METRICS, MODELS, DefaultScope
from mmengine.runner import Runner
from mmengine.visualization import Visualizer


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_samples, list):
            data_samples = torch.stack(data_samples)
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_samples - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs


class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_samples=self.label[index])


class ToyMetric(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class RunnerTestCase(TestCase):
    """A test case to build runner easily.

    `RunnerTestCase` will do the following things:

    1. Registers a toy model, a toy metric, and a toy dataset, which can be
       used to run the `Runner` successfully.
    2. Provides epoch based and iteration based cfg to build runner.
    3. Provides `build_runner` method to build runner easily.
    4. Clean the global variable used by the runner.
    """
    dist_cfg = dict(
        MASTER_ADDR='127.0.0.1',
        MASTER_PORT=29600,
        RANK='0',
        WORLD_SIZE='1',
        LOCAL_RANK='0')

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        # Prevent from registering module with the same name by other unit
        # test. These registries will be cleared in `tearDown`
        MODELS.register_module(module=ToyModel, force=True)
        METRICS.register_module(module=ToyMetric, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        epoch_based_cfg = dict(
            work_dir=self.temp_dir.name,
            model=dict(type='ToyModel'),
            train_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=[dict(type='ToyMetric')],
            test_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            test_evaluator=[dict(type='ToyMetric')],
            optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.1)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(),
            test_cfg=dict(),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            custom_hooks=[],
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
            experiment_name='test1')
        self.epoch_based_cfg = Config(epoch_based_cfg)

        # prepare iter based cfg.
        self.iter_based_cfg: Config = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='InfiniteSampler', shuffle=True),
            batch_size=3,
            num_workers=0)
        self.iter_based_cfg.log_processor = dict(by_epoch=False)

        self.iter_based_cfg.train_cfg = dict(by_epoch=False, max_iters=12)
        self.iter_based_cfg.default_hooks = dict(
            logger=dict(type='LoggerHook', interval=1),
            checkpoint=dict(
                type='CheckpointHook', interval=12, by_epoch=False))

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        Visualizer._instance_dict.clear()
        DefaultScope._instance_dict.clear()
        MessageHub._instance_dict.clear()
        MODELS.module_dict.pop('ToyModel', None)
        METRICS.module_dict.pop('ToyMetric', None)
        DATASETS.module_dict.pop('ToyDataset', None)
        self.temp_dir.cleanup()
        if is_distributed():
            destroy_process_group()

    def build_runner(self, cfg: Config):
        cfg.experiment_name = self.experiment_name
        runner = Runner.from_cfg(cfg)
        return runner

    @property
    def experiment_name(self):
        # Since runners could be built too fast to have a unique experiment
        # name(timestamp is the same), here we use uuid to make sure each
        # runner has the unique experiment name.
        return f'{self._testMethodName}_{time.time()} + ' \
               f'{uuid4()}'

    def setup_dist_env(self):
        self.dist_cfg['MASTER_PORT'] += 1
        os.environ['MASTER_PORT'] = str(self.dist_cfg['MASTER_PORT'])
        os.environ['MASTER_ADDR'] = self.dist_cfg['MASTER_ADDR']
        os.environ['RANK'] = self.dist_cfg['RANK']
        os.environ['WORLD_SIZE'] = self.dist_cfg['WORLD_SIZE']
        os.environ['LOCAL_RANK'] = self.dist_cfg['LOCAL_RANK']

    def clear_work_dir(self):
        logging.shutdown()
        for filename in os.listdir(self.temp_dir.name):
            filepath = os.path.join(self.temp_dir.name, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath)
