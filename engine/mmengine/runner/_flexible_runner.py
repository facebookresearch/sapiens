# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os.path as osp
import pickle
import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch.nn as nn
from torch.utils.data import DataLoader

import mmengine
from mmengine._strategy import BaseStrategy
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.dist import get_rank, infer_launcher, master_only
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, print_log
from mmengine.optim import OptimWrapper, OptimWrapperDict, _ParamScheduler
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, RUNNERS,
                               STRATEGIES, VISUALIZERS, DefaultScope)
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.visualization import Visualizer
from .base_loop import BaseLoop
from .checkpoint import find_latest_checkpoint
from .log_processor import LogProcessor
from .loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from .priority import Priority, get_priority

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]


@RUNNERS.register_module()
class FlexibleRunner:
    """A training helper for PyTorch.

    Runner object can be built from config by ``runner = Runner.from_cfg(cfg)``
    where the ``cfg`` usually contains training, validation, and test-related
    configurations to build corresponding components. We usually use the
    same config to launch training, testing, and validation tasks. However,
    only some of these components are necessary at the same time, e.g.,
    testing a model does not need training or validation-related components.

    To avoid repeatedly modifying config, the construction of ``Runner`` adopts
    lazy initialization to only initialize components when they are going to be
    used. Therefore, the model is always initialized at the beginning, and
    training, validation, and, testing related components are only initialized
    when calling ``runner.train()``, ``runner.val()``, and ``runner.test()``,
    respectively.

    Warning:
        This is an experimental feature, and its interface is subject to
        change.

    Args:
        model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
            a dict used for build a model.

    Kwargs:
        work_dir (str, optional): The working directory to save checkpoints.
            The logs will be saved in the subdirectory of `work_dir` named
            :attr:`timestamp`. Defaults to 'work_dir'.
        experiment_name (str, optional): Name of current experiment. If not
            specified, timestamp will be used as ``experiment_name``.
            Defaults to None.
        train_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping training steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        optim_wrapper (OptimWrapper or dict, optional):
            Computing gradient of model parameters. If specified,
            :attr:`train_dataloader` should also be specified. If automatic
            mixed precision or gradient accmulation
            training is required. The type of ``optim_wrapper`` should be
            AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
            examples. Defaults to None.
        param_scheduler (_ParamScheduler or dict or list, optional):
            Parameter scheduler for updating optimizer parameters. If
            specified, :attr:`optimizer` should also be specified.
            Defaults to None.
            See :meth:`build_param_scheduler` for examples.
        train_cfg (dict, optional): A dict to build a training loop. If it does
            not provide "type" key, it should contain "by_epoch" to decide
            which type of training loop :class:`EpochBasedTrainLoop` or
            :class:`IterBasedTrainLoop` should be used. If ``train_cfg``
            specified, :attr:`train_dataloader` should also be specified.
            Defaults to None. See :meth:`build_train_loop` for more details.
        val_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping validation steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        val_evaluator (Evaluator or dict or list, optional): A evaluator object
            used for computing metrics for validation. It can be a dict or a
            list of dict to build a evaluator. If specified,
            :attr:`val_dataloader` should also be specified. Defaults to None.
        val_cfg (dict, optional): A dict to build a validation loop. If it does
            not provide "type" key, :class:`ValLoop` will be used by default.
            If ``val_cfg`` specified, :attr:`val_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
        test_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping test steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
            Defaults to None. See :meth:`build_val_loop` for more details.
        test_evaluator (Evaluator or dict or list, optional): A evaluator
            object used for computing metrics for test steps. It can be a dict
            or a list of dict to build a evaluator. If specified,
            :attr:`test_dataloader` should also be specified. Defaults to None.
        test_cfg (dict, optional): A dict to build a test loop. If it does
            not provide "type" key, :class:`TestLoop` will be used by default.
            If ``test_cfg`` specified, :attr:`test_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_test_loop` for more details.
        strategy (BaseStrategy or dict, optional): A strategy object or a dict
            to build a strategy. Defaults to None. If not specified, the
            strategy will be inferred automatically.
        auto_scale_lr (dict, Optional): Config to scale the learning rate
            automatically. It includes ``base_batch_size`` and ``enable``.
            ``base_batch_size`` is the batch size that the optimizer lr is
            based on. ``enable`` is the switch to turn on and off the feature.
        default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
            execute default actions like updating model parameters and saving
            checkpoints. Default hooks are ``OptimizerHook``,
            ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
            ``CheckpointHook``. Defaults to None.
            See :meth:`register_default_hooks` for more details.
        custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
            custom actions like visualizing images processed by pipeline.
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. If the ``model`` argument is a dict
            and doesn't contain the key ``data_preprocessor``, set the argument
            as the ``data_preprocessor`` of the ``model`` dict.
            Defaults to None.
        load_from (str, optional): The checkpoint file to load from.
            Defaults to None.
        resume (bool): Whether to resume training. Defaults to False. If
            ``resume`` is True and ``load_from`` is None, automatically to
            find latest checkpoint from ``work_dir``. If not found, resuming
            does nothing.
        launcher (str, optional): Way to launcher multi-process. Supported
            launchers are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is
            provided, non-distributed environment will be launched.
            If launcher is None, the launcher will be inferred according some
            specified environments. Defaults to None.
        env_cfg (dict): A dict used for setting environment. Defaults to
            dict(dist_cfg=dict(backend='nccl')).
        log_processor (dict, optional): A processor to format logs. Defaults to
            None.
        log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        visualizer (Visualizer or dict, optional): A Visualizer object or a
            dict build Visualizer object. Defaults to None. If not
            specified, default config will be used.
        default_scope (str): Used to reset registries location.
            Defaults to "mmengine".
        randomness (dict): Some settings to make the experiment as reproducible
            as possible like seed and deterministic.
            Defaults to ``dict(seed=None)``. If seed is None, a random number
            will be generated and it will be broadcasted to all other processes
            if in distributed environment. If ``cudnn_benchmark`` is
            ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
            ``randomness``, the value of ``torch.backends.cudnn.benchmark``
            will be ``False`` finally.
        compile (bool or dict, optional): Whether to enable ``torch.compile``.
            Defaults to False.
        cfg (dict or Configdict or :obj:`Config`, optional): Full config.
            Defaults to None.

    Note:
        Since PyTorch 2.0.0, you can enable ``torch.compile`` by passing in
        `compile = True`. If you want to control compile options, you
        can pass a dict, e.g. ``cfg.compile = dict(backend='eager')``.
        Refer to `PyTorch API Documentation <https://pytorch.org/docs/
        master/generated/torch.compile.html#torch.compile>`_ for more valid
        options.

    Examples:
        >>> from mmengine.runner import Runner
        >>> cfg = dict(
        >>>     model=dict(type='ToyModel'),
        >>>     work_dir='path/of/work_dir',
        >>>     train_dataloader=dict(
        >>>     dataset=dict(type='ToyDataset'),
        >>>     sampler=dict(type='DefaultSampler', shuffle=True),
        >>>     batch_size=1,
        >>>     num_workers=0),
        >>>     val_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>        batch_size=1,
        >>>        num_workers=0),
        >>>     test_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>         batch_size=1,
        >>>         num_workers=0),
        >>>     auto_scale_lr=dict(base_batch_size=16, enable=False),
        >>>     optim_wrapper=dict(type='OptimizerWrapper', optimizer=dict(
        >>>         type='SGD', lr=0.01)),
        >>>     param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
        >>>     val_evaluator=dict(type='ToyEvaluator'),
        >>>     test_evaluator=dict(type='ToyEvaluator'),
        >>>     train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1),
        >>>     val_cfg=dict(),
        >>>     test_cfg=dict(),
        >>>     custom_hooks=[],
        >>>     default_hooks=dict(
        >>>         timer=dict(type='IterTimerHook'),
        >>>         checkpoint=dict(type='CheckpointHook', interval=1),
        >>>         logger=dict(type='LoggerHook'),
        >>>         optimizer=dict(type='OptimizerHook', grad_clip=False),
        >>>         param_scheduler=dict(type='ParamSchedulerHook')),
        >>>     launcher='none',
        >>>     env_cfg=dict(dist_cfg=dict(backend='nccl')),
        >>>     log_processor=dict(window_size=20),
        >>>     visualizer=dict(type='Visualizer',
        >>>     vis_backends=[dict(type='LocalVisBackend',
        >>>                        save_dir='temp_dir')])
        >>>    )
        >>> runner = Runner.from_cfg(cfg)
        >>> runner.train()
        >>> runner.test()
    """
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        *,
        work_dir: str = 'work_dirs',
        experiment_name: Optional[str] = None,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        train_cfg: Optional[Dict] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        val_cfg: Optional[Dict] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_cfg: Optional[Dict] = None,
        strategy: Optional[Union[BaseStrategy, Dict]] = None,
        auto_scale_lr: Optional[Dict] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: Union[str, bool] = False,
        launcher: Optional[str] = None,
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: Optional[str] = 'mmengine',
        randomness: Dict = dict(seed=None),
        compile: Union[bool, Dict] = False,
        cfg: Optional[ConfigType] = None,
    ):
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = model

        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self._auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        if not isinstance(compile, bool) and not isinstance(compile, dict):
            raise TypeError(
                f'compile should be a bool or dict, but got {type(compile)}')
        self._compile = compile

        if isinstance(resume, str) and load_from is not None:
            raise ValueError('If resume is a str, load_from should be None.')
        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        if launcher is None:
            launcher = infer_launcher()

        if experiment_name is None and self.cfg.filename is not None:
            experiment_name = osp.splitext(osp.basename(self.cfg.filename))[0]

        self._randomness_cfg = randomness
        self.strategy = self.build_strategy(
            strategy,
            launcher=launcher,
            randomness=randomness,
            env_cfg=env_cfg,
            experiment_name=experiment_name,
            log_level=log_level,
        )

        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self.experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope
        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)

        # Collect and log environment information.
        self._log_env()

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        self.dump_config()

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'FlexibleRunner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg.get('work_dir', 'work_dirs'),
            experiment_name=cfg.get('experiment_name'),
            train_dataloader=cfg.get('train_dataloader'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            train_cfg=cfg.get('train_cfg'),
            val_dataloader=cfg.get('val_dataloader'),
            val_evaluator=cfg.get('val_evaluator'),
            val_cfg=cfg.get('val_cfg'),
            test_dataloader=cfg.get('test_dataloader'),
            test_evaluator=cfg.get('test_evaluator'),
            test_cfg=cfg.get('test_cfg'),
            strategy=cfg.get('strategy'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            cfg=cfg,
        )

        return runner

    @property
    def experiment_name(self):
        """str: Name of experiment."""
        return self.strategy.experiment_name

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def work_dir(self):
        """str: The working directory to save checkpoints and logs."""
        return self._work_dir

    @property
    def log_dir(self):
        return self.strategy.log_dir

    @property
    def logger(self):
        return self.strategy.logger

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.max_epochs
        else:
            return 0

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.max_iters
        else:
            return 0

    @property
    def epoch(self):
        """int: Current epoch."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.epoch
        else:
            return 0

    @property
    def iter(self):
        """int: Current iteration."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.iter
        else:
            return 0

    @property
    def distributed(self):
        """bool: Whether current environment is distributed."""
        return self.strategy.distributed

    @property
    def rank(self):
        """int: Rank of current process."""
        return self.strategy.rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job."""
        return self.strategy.world_size

    @property
    def deterministic(self):
        """int: Whether cudnn to select deterministic algorithms."""
        return self._deterministic

    @property
    def seed(self):
        """int: A number to set random modules."""
        return self.strategy.seed

    @property
    def timestamp(self):
        """str: Timestamp when creating experiment."""
        return self.strategy.timestamp

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def train_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._train_loop, BaseLoop) or self._train_loop is None:
            return self._train_loop
        else:
            self._train_loop = self.build_train_loop(self._train_loop)
            return self._train_loop

    @property
    def val_loop(self):
        """:obj:`BaseLoop`: A loop to run validation."""
        if isinstance(self._val_loop, BaseLoop) or self._val_loop is None:
            return self._val_loop
        else:
            self._val_loop = self.build_val_loop(self._val_loop)
            return self._val_loop

    @property
    def test_loop(self):
        """:obj:`BaseLoop`: A loop to run testing."""
        if isinstance(self._test_loop, BaseLoop) or self._test_loop is None:
            return self._test_loop
        else:
            self._test_loop = self.build_test_loop(self._test_loop)
            return self._test_loop

    @property
    def train_dataloader(self):
        """The data loader for training."""
        return self.train_loop.dataloader

    @property
    def val_dataloader(self):
        """The data loader for validation."""
        return self.val_loop.dataloader

    @property
    def test_dataloader(self):
        """The data loader for testing."""
        return self.test_loop.dataloader

    @property
    def val_evaluator(self):
        """:obj:`Evaluator`: An evaluator for validation."""
        return self.val_loop.evaluator

    @property
    def test_evaluator(self):
        """:obj:`Evaluator`: An evaluator for testing."""
        return self.test_loop.evaluator

    @property
    def val_interval(self):
        """int: Interval to run validation during training."""
        return self.train_loop.val_interval

    @property
    def val_begin(self):
        """int: The epoch/iteration to start running validation during
        training."""
        return self.train_loop.val_begin

    def build_strategy(
        self,
        strategy: Optional[Union[BaseStrategy, Dict]] = None,
        launcher: str = 'none',
        randomness: Optional[dict] = None,
        env_cfg: dict = dict(dist_cfg=dict(backend='nccl')),
        experiment_name: Optional[str] = None,
        log_level: Optional[str] = None,
    ) -> BaseStrategy:
        """Build a strategy.

        Args:
            strategy (BaseStrategy, optional): A strategy object or dict to
                build the strategy. Defaults to None.

        Returns:
            BaseStrategy: A strategy object.
        """
        if isinstance(strategy, BaseStrategy):
            strategy_obj = strategy
        else:
            if launcher == 'none':
                if strategy is None:
                    strategy = dict(type='SingleDeviceStrategy')
            else:
                if strategy is None:
                    strategy = dict(type='DDPStrategy')

            assert isinstance(strategy, dict)

            # train_micro_batch_size_per_gpu is required by DeepSpeed
            if isinstance(strategy['type'], str):
                strategy_name = strategy['type']
            else:
                strategy_name = strategy['type'].__name__
            if strategy_name == 'DeepSpeedStrategy':
                if self._train_dataloader is None:
                    strategy['train_micro_batch_size_per_gpu'] = 1
                else:
                    strategy['train_micro_batch_size_per_gpu'] = \
                        _get_batch_size(self._train_dataloader)

            ## for FSDPStrategy, we need to set micro batch size per gpu
            if strategy_name == 'FSDPStrategy':
                if self._train_dataloader is None:
                    strategy['train_micro_batch_size_per_gpu'] = 1
                else:
                    strategy['train_micro_batch_size_per_gpu'] = \
                        _get_batch_size(self._train_dataloader)

            strategy.setdefault('work_dir', self._work_dir)
            strategy.setdefault('experiment_name', experiment_name)
            strategy.setdefault('auto_scale_lr', self._auto_scale_lr)

            env_kwargs = dict(
                launcher=launcher,
                randomness=randomness,
                **env_cfg,
            )
            strategy.setdefault('env_kwargs', env_kwargs)

            log_kwargs = dict(log_level=log_level)
            strategy.setdefault('log_kwargs', log_kwargs)

            strategy_obj = STRATEGIES.build(strategy)

        return strategy_obj

    def build_message_hub(
        self,
        message_hub: Optional[Dict] = None,
    ) -> MessageHub:
        """Build a global asscessable MessageHub.

        Args:
            message_hub (dict, optional): A dict to build MessageHub object.
                If not specified, default config will be used to build
                MessageHub object. Defaults to None.

        Returns:
            MessageHub: A MessageHub object build from ``message_hub``.
        """
        if message_hub is None:
            message_hub = dict(name=self.experiment_name)
        elif isinstance(message_hub, dict):
            # ensure message_hub containing name key
            message_hub.setdefault('name', self.experiment_name)
        else:
            raise TypeError(
                f'message_hub should be dict or None, but got {message_hub}')

        return MessageHub.get_instance(**message_hub)

    def build_visualizer(
        self,
        visualizer: Optional[Union[Visualizer, Dict]] = None,
    ) -> Visualizer:
        """Build a global asscessable Visualizer.

        Args:
            visualizer (Visualizer or dict, optional): A Visualizer object
                or a dict to build Visualizer object. If ``visualizer`` is a
                Visualizer object, just returns itself. If not specified,
                default config will be used to build Visualizer object.
                Defaults to None.

        Returns:
            Visualizer: A Visualizer object build from ``visualizer``.
        """
        if visualizer is None:
            visualizer = dict(
                name=self.experiment_name,
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=self.log_dir)
            return Visualizer.get_instance(**visualizer)

        if isinstance(visualizer, Visualizer):
            return visualizer

        if isinstance(visualizer, dict):
            # ensure visualizer containing name key
            visualizer.setdefault('name', self.experiment_name)
            visualizer.setdefault('save_dir', self.log_dir)
            return VISUALIZERS.build(visualizer)
        else:
            raise TypeError(
                'visualizer should be Visualizer object, a dict or None, '
                f'but got {visualizer}')

    def build_evaluator(
        self,
        evaluator: Union[Dict, List, Evaluator],
    ) -> Evaluator:
        """Build evaluator.

        Examples of ``evaluator``::

            # evaluator could be a built Evaluator instance
            evaluator = Evaluator(metrics=[ToyMetric()])

            # evaluator can also be a list of dict
            evaluator = [
                dict(type='ToyMetric1'),
                dict(type='ToyEvaluator2')
            ]

            # evaluator can also be a list of built metric
            evaluator = [ToyMetric1(), ToyMetric2()]

            # evaluator can also be a dict with key metrics
            evaluator = dict(metrics=ToyMetric())
            # metric is a list
            evaluator = dict(metrics=[ToyMetric()])

        Args:
            evaluator (Evaluator or dict or list): An Evaluator object or a
                config dict or list of config dict used to build an Evaluator.

        Returns:
            Evaluator: Evaluator build from ``evaluator``.
        """
        if isinstance(evaluator, Evaluator):
            return evaluator
        elif isinstance(evaluator, dict):
            # if `metrics` in dict keys, it means to build customized evalutor
            if 'metrics' in evaluator:
                evaluator.setdefault('type', 'Evaluator')
                return EVALUATOR.build(evaluator)
            # otherwise, default evalutor will be built
            else:
                return Evaluator(evaluator)  # type: ignore
        elif isinstance(evaluator, list):
            # use the default `Evaluator`
            return Evaluator(evaluator)  # type: ignore
        else:
            raise TypeError(
                'evaluator should be one of dict, list of dict, and Evaluator'
                f', but got {evaluator}')

    @staticmethod
    def build_dataloader(
        dataloader: Union[DataLoader, Dict],
        seed: Optional[int] = None,
        diff_rank_seed: bool = False,
    ) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks. Defaults to False.

        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]
        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                              **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop('type')
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                'collate_fn should be a dict or callable object, but got '
                f'{collate_fn_cfg}')
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build training loop.

        Examples of ``loop``::

            # `EpochBasedTrainLoop` will be used
            loop = dict(by_epoch=True, max_epochs=3)

            # `IterBasedTrainLoop` will be used
            loop = dict(by_epoch=False, max_epochs=3)

            # custom training loop
            loop = dict(type='CustomTrainLoop', max_epochs=3)

        Args:
            loop (BaseLoop or dict): A training loop or a dict to build
                training loop. If ``loop`` is a training loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self, dataloader=self._train_dataloader))
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
        return loop  # type: ignore

    def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build validation loop.

        Examples of ``loop``:

            # `ValLoop` will be used
            loop = dict()

            # custom validation loop
            loop = dict(type='CustomValLoop')

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._val_dataloader,
                    evaluator=self._val_evaluator))
        else:
            loop = ValLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._val_dataloader,
                evaluator=self._val_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build test loop.

        Examples of ``loop``::

            # `TestLoop` will be used
            loop = dict()

            # custom test loop
            loop = dict(type='CustomTestLoop')

        Args:
            loop (BaseLoop or dict): A test loop or a dict to build test loop.
                If ``loop`` is a test loop object, just returns itself.

        Returns:
            :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)  # type: ignore

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._test_dataloader,
                    evaluator=self._test_evaluator))
        else:
            loop = TestLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._test_dataloader,
                evaluator=self._test_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_log_processor(
        self,
        log_processor: Union[LogProcessor, Dict],
    ) -> LogProcessor:
        """Build test log_processor.

        Examples of ``log_processor``:

            # `LogProcessor` will be used
            log_processor = dict()

            # custom log_processor
            log_processor = dict(type='CustomLogProcessor')

        Args:
            log_processor (LogProcessor or dict): A log processor or a dict
            to build log processor. If ``log_processor`` is a log processor
            object, just returns itself.

        Returns:
            :obj:`LogProcessor`: Log processor object build from
            ``log_processor_cfg``.
        """
        if isinstance(log_processor, LogProcessor):
            return log_processor
        elif not isinstance(log_processor, dict):
            raise TypeError(
                'log processor should be a LogProcessor object or dict, but'
                f'got {log_processor}')

        log_processor_cfg = copy.deepcopy(log_processor)  # type: ignore

        if 'type' in log_processor_cfg:
            log_processor = LOG_PROCESSORS.build(log_processor_cfg)
        else:
            log_processor = LogProcessor(**log_processor_cfg)  # type: ignore

        return log_processor  # type: ignore

    def get_hooks_info(self) -> str:
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def load_or_resume(self):
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        if not self._resume and self._load_from is None:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if isinstance(self._resume, str):
            resume_from = self._resume
        elif self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        if resume_from is not None:
            self.resume(resume_from, map_location="cpu")
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True

    def train(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore

        compile: Union[dict, bool] = False
        if isinstance(self._compile, bool):
            if self._compile:
                compile = dict(target='train_step')
        else:
            compile = copy.copy(self._compile)
            compile.setdefault('target', 'train_step')

        dispatch_kwargs = dict(
            epoch_length=len(self.train_dataloader),
            max_epochs=self.max_epochs,
            max_iters=self.max_iters,
        )
        self.strategy.prepare(
            self.model,
            optim_wrapper=self.optim_wrapper,
            param_scheduler=self.param_schedulers,
            compile=compile,
            dispatch_kwargs=dispatch_kwargs,
        )

        self.model = self.strategy.model
        self.optim_wrapper = self.strategy.optim_wrapper  # type: ignore
        if self.param_schedulers is not None:
            self.param_schedulers = self.strategy.param_schedulers

        self.load_or_resume()

        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return model

    def val(self) -> dict:
        """Launch validation.

        Returns:
            dict: A dict of metrics on validation set.
        """
        if self._val_loop is None:
            raise RuntimeError(
                '`self._val_loop` should not be None when calling val method.'
                'Please provide `val_dataloader`, `val_cfg` and '
                '`val_evaluator` arguments when initializing runner.')

        self._val_loop = self.build_val_loop(self._val_loop)  # type: ignore

        dispatch_kwargs = dict(
            init_weights_for_test_or_val=self.cfg.get(
                'init_weights_for_test_or_val', True))
        self.strategy.prepare(self.model, dispatch_kwargs=dispatch_kwargs)
        self.model = self.strategy.model

        self.load_or_resume()

        self.call_hook('before_run')
        metrics = self.val_loop.run()  # type: ignore
        self.call_hook('after_run')

        return metrics

    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore
        dispatch_kwargs = dict(
            init_weights_for_test_or_val=self.cfg.get(
                'init_weights_for_test_or_val', True))
        self.strategy.prepare(self.model, dispatch_kwargs=dispatch_kwargs)
        self.model = self.strategy.model

        self.load_or_resume()

        self.call_hook('before_run')
        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')

        return metrics

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments passed to hook.
        """
        for hook in self._hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from e

    def register_hook(
        self,
        hook: Union[Hook, Dict],
        priority: Optional[Union[str, int, Priority]] = None,
    ) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Priority of hook will be decided with the following priority:

        - ``priority`` argument. If ``priority`` is given, it will be priority
          of hook.
        - If ``hook`` argument is a dict and ``priority`` in it, the priority
          will be the value of ``hook['priority']``.
        - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
          is an instance of ``hook``, the priority will be ``hook.priority``.

        Args:
            hook (:obj:`Hook` or dict): The hook to be registered.
            priority (int or str or :obj:`Priority`, optional): Hook priority.
                Lower value means higher priority.
        """
        if not isinstance(hook, (Hook, dict)):
            raise TypeError(
                f'hook should be an instance of Hook or dict, but got {hook}')

        _priority = None
        if isinstance(hook, dict):
            if 'priority' in hook:
                _priority = hook.pop('priority')

            hook_obj = HOOKS.build(hook)
        else:
            hook_obj = hook

        if priority is not None:
            hook_obj.priority = priority
        elif _priority is not None:
            hook_obj.priority = _priority

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook_obj.priority) >= get_priority(
                    self._hooks[i].priority):
                self._hooks.insert(i + 1, hook_obj)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook_obj)

    def register_default_hooks(
        self,
        hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
    ) -> None:
        """Register default hooks into hook list.

        ``hooks`` will be registered into runner to execute some default
        actions like updating model parameters or saving checkpoints.

        Default hooks and their priorities:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | RuntimeInfoHook      | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (50)             |
        +----------------------+-------------------------+
        | DistSamplerSeedHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | LoggerHook           | BELOW_NORMAL (60)       |
        +----------------------+-------------------------+
        | ParamSchedulerHook   | LOW (70)                |
        +----------------------+-------------------------+
        | CheckpointHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+

        If ``hooks`` is None, above hooks will be registered by
        default::

            default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
            )

        If not None, ``hooks`` will be merged into ``default_hooks``.
        If there are None value in default_hooks, the corresponding item will
        be popped from ``default_hooks``::

            hooks = dict(timer=None)

        The final registered default hooks will be :obj:`RuntimeInfoHook`,
        :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`,
        :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

        Args:
            hooks (dict[str, Hook or dict], optional): Default hooks or configs
                to be registered.
        """
        default_hooks: dict = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
        )
        if hooks is not None:
            for name, hook in hooks.items():
                if name in default_hooks and hook is None:
                    # remove hook from _default_hooks
                    default_hooks.pop(name)
                else:
                    assert hook is not None
                    default_hooks[name] = hook

        for hook in default_hooks.values():
            self.register_hook(hook)

    def register_custom_hooks(self, hooks: List[Union[Hook, Dict]]) -> None:
        """Register custom hooks into hook list.

        Args:
            hooks (list[Hook | dict]): List of hooks or configs to be
                registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hooks(
        self,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
    ) -> None:
        """Register default hooks and custom hooks into hook list.

        Args:
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
                to execute default actions like updating model parameters and
                saving checkpoints.  Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
        self.register_default_hooks(default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)

    def resume(
        self,
        filename: str,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: Union[str, Callable] = 'default',
    ) -> None:
        """Resume model from checkpoint.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
        """

        def callback(checkpoint):
            self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        checkpoint = self.strategy.resume(
            filename,
            resume_optimizer=resume_optimizer,
            resume_param_scheduler=resume_param_scheduler,
            map_location=map_location,
            callback=callback,
        )

        self.train_loop._epoch = checkpoint['meta']['epoch']
        self.train_loop._iter = checkpoint['meta']['iter']

        # check whether the number of GPU used for current experiment
        # is consistent with resuming from checkpoint
        if 'config' in checkpoint['meta']:
            config = mmengine.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
                    and len(previous_gpu_ids) != self.world_size):
                # TODO, should we modify the iteration?
                self.logger.info(
                    'Number of GPU used for current experiment is not '
                    'consistent with resuming from checkpoint')
                if (self._auto_scale_lr is None
                        or not self._auto_scale_lr.get('enable', False)):
                    raise RuntimeError(
                        'Cannot automatically rescale lr in resuming. Please '
                        'make sure the number of GPU is consistent with the '
                        'previous training state resuming from the checkpoint '
                        'or set `enable` in `auto_scale_lr to False.')

        resumed_dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        dataset_meta = getattr(self.train_dataloader.dataset, 'metainfo', None)

        # `resumed_dataset_meta` and `dataset_meta` could be object like
        # np.ndarray, which cannot be directly judged as equal or not,
        # therefore we just compared their dumped results.
        if pickle.dumps(resumed_dataset_meta) != pickle.dumps(dataset_meta):
            self.logger.warning(
                'The dataset metainfo from the resumed checkpoint is '
                'different from the current training dataset, please '
                'check the correctness of the checkpoint or the training '
                'dataset.')

        self.message_hub.load_state_dict(checkpoint['message_hub'])

        self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

    def load_checkpoint(self,
                        filename: str,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """

        def callback(checkpoint):
            self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        self.strategy.load_checkpoint(
            filename,
            map_location=map_location,
            strict=strict,
            revise_keys=revise_keys,
            callback=callback)

    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: dict = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Whether the scheduled momentum is updated by
                epochs. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch, iter=self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text, experiment_name=self.experiment_name)

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        checkpoint = {
            'meta': meta,
            'message_hub': self.message_hub.state_dict()
        }

        def callback(checkpoint):
            self.call_hook('before_save_checkpoint', checkpoint=checkpoint)

        self.strategy.save_checkpoint(
            filename=filepath,
            save_optimizer=save_optimizer,
            save_param_scheduler=save_param_scheduler,
            extra_ckpt=checkpoint,
            callback=callback,
        )

    @master_only
    def dump_config(self) -> None:
        """Dump config to `work_dir`."""
        if self.cfg.filename is not None:
            filename = osp.basename(self.cfg.filename)
        else:
            filename = f'{self.timestamp}.py'
        self.cfg.dump(osp.join(self.work_dir, filename))

    def _log_env(self) -> None:
        """Logging environment information of the current task.

        Args:
            env_cfg (dict): The environment config of the runner.
        """
        # Collect and log environment information.
        system_env, runtime_env = self.strategy.collect_env()

        env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
                                            for k, v in system_env.items())
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        self.logger.info('\n' + dash_line + '\nSystem environment:' +
                         env_info + '\n'
                         '\nRuntime environment:' + runtime_env_info + '\n' +
                         dash_line + '\n')

        if self.cfg._cfg_dict:
            self.logger.info(f'Config:\n{self.cfg.pretty_text}')


def _get_batch_size(dataloader):
    if isinstance(dataloader, dict):
        if 'batch_size' in dataloader:
            return dataloader['batch_size']
        elif ('batch_sampler' in dataloader
              and 'batch_size' in dataloader['batch_sampler']):
            return dataloader['batch_sampler']['batch_size']
        else:
            raise ValueError('Please set batch_size in `Dataloader` or '
                             '`batch_sampler`')
    elif isinstance(dataloader, DataLoader):
        return dataloader.batch_sampler.batch_size
    else:
        raise ValueError('dataloader should be a dict or a Dataloader '
                         f'instance, but got {type(dataloader)}')
