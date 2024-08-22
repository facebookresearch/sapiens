# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
import platform
import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dist import (broadcast, get_dist_info, infer_launcher,
                           is_distributed)
from mmengine.logging import MMLogger
from mmengine.model.wrappers import is_model_wrapper
from mmengine.optim import (BaseOptimWrapper, OptimWrapperDict,
                            _ParamScheduler, build_optim_wrapper)
from mmengine.registry import MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)

ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]


class BaseStrategy(metaclass=ABCMeta):
    """Base class for all strategies.

    In the process of supporting FSDP, DeepSpeed, and ColossalAI, the
    scalability of the Runner faced challenges, which led to the redefinition
    of the Runner's responsibilities. The Strategy abstraction was split out,
    which is responsible for constructing, initializing, and saving/loading
    the state of training components such as models, optimizers, and parameter
    schedulers.

    Warning:
        This is an experimental feature, and its interface is subject to
        change.

    Keyword Args:
        work_dir (str): The working directory to save checkpoints. The logs
            will be saved in the subdirectory of `work_dir` named
            :attr:`timestamp`. Defaults to 'work_dirs'.
        experiment_name (str, optional): Name of current experiment. If not
            specified, timestamp will be used as :attr:`experiment_name`.
            Defaults to None.
        env_kwargs (dict, optional): Environment config passed in
            :meth:`setup_env`. Defaults to None.
        log_kwargs (dict, optional): Logger config passed in
            :meth:`build_logger`. Defaults to None.
        auto_scale_lr (dict, Optional): Config to scale the learning rate
            automatically. It includes ``base_batch_size`` and ``enable``.
            ``base_batch_size`` is the batch size that the optimizer lr is
            based on. ``enable`` is the switch to turn on and off the feature.
    """
    model: nn.Module
    optim_wrapper: BaseOptimWrapper
    param_schedulers: ParamSchedulerType

    def __init__(
        self,
        *,
        work_dir: str = 'work_dirs',
        experiment_name: Optional[str] = None,
        env_kwargs: Optional[dict] = None,
        log_kwargs: Optional[dict] = None,
        auto_scale_lr: Optional[dict] = None,
    ):
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        self._env_kwargs = env_kwargs or {}
        self._setup_env(**self._env_kwargs)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self.timestamp}'
        else:
            self._experiment_name = self.timestamp

        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)

        log_kwargs = log_kwargs or {}
        self.logger = self.build_logger(**log_kwargs)

        self._auto_scale_lr = auto_scale_lr

        self.dispatch_kwargs: dict = {}
        self._prepared = False

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def launcher(self):
        return self._launcher

    @property
    def distributed(self):
        return self._distributed

    @property
    def seed(self):
        return self._seed

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def randomness(self):
        return self._randomness

    @abstractmethod
    def prepare(
        self,
        model: Union[nn.Module, dict],
        *,
        optim_wrapper: Union[BaseOptimWrapper, dict, None] = None,
        param_scheduler: Union[_ParamScheduler, Dict, List, None] = None,
        compile: Union[dict, bool] = False,
        dispatch_kwargs: Optional[dict] = None,
    ):
        """Prepare model and some components.

        Args:
            model (:obj:`torch.nn.Module` or dict): The model to be run. It
                can be a dict used for building a model.

        Keyword Args:
            optim_wrapper (BaseOptimWrapper or dict, optional): Computing the
                gradient of model parameters and updating them.
                Defaults to None.
                See :meth:`build_optim_wrapper` for examples.
            param_scheduler (_ParamScheduler or dict or list, optional):
                Parameter scheduler for updating optimizer parameters. If
                specified, :attr:`optim_wrapper` should also be specified.
                Defaults to None.
                See :meth:`build_param_scheduler` for examples.
            compile (dict, optional): Config to compile model.
                Defaults to False. Requires PyTorch>=2.0.
            dispatch_kwargs (dict, optional): Kwargs to be passed to other
                methods of Strategy. Defaults to None.
        """

    def _setup_env(
            self,
            *,
            launcher: Optional[str] = None,
            cudnn_benchmark: bool = False,
            mp_cfg: Optional[dict] = None,
            dist_cfg: Optional[dict] = None,
            resource_limit: int = 4096,
            randomness: dict = dict(seed=None),
    ):
        """Setup environment.

        This method will do the following things:

        1. setup multi-processing
        2. setup distributed
        3. set random seed

        Keyword Args:
            launcher (str, optional): Way to launcher multi-process. Supported
                launchers are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none'
                is provided, non-distributed environment will be launched.
                If launcher is None, the launcher will be inferred according
                some specified environments. Defaults to None.
            cudnn_benchmark (bool): Whether to enable cudnn benchmark.
                Defaults to False.
            mp_cfg (dict, optional): Multi-processing config. Defaults to None.
            dist_cfg (dict, optional): Distributed config. Defaults to None.
            resource_limit (int): Resource limit. Defaults to 4096.
            randomness (dict): Some settings to make the experiment as
                reproducible as possible like seed and deterministic.
                Defaults to ``dict(seed=None)``. If seed is None, a random
                number will be generated and it will be broadcasted to all
                other processes if in distributed environment.
                If ``cudnn_benchmark`` is ``True`` in but ``deterministic`` is
                ``True`` in ``randomness``, the value of
                ``torch.backends.cudnn.benchmark`` will be ``False`` finally.
        """
        if launcher is None:
            launcher = infer_launcher()

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        if cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        mp_cfg = mp_cfg if mp_cfg is not None else {}
        set_multi_processing(**mp_cfg, distributed=self._distributed)

        # init distributed env first, since logger depends on the dist info.
        if self._distributed and not is_distributed():
            dist_cfg = dist_cfg if dist_cfg is not None else {}
            self._setup_distributed(launcher, **dist_cfg)

        self._rank, self._world_size = get_dist_info()

        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        # broadcast timestamp from 0 process to other processes
        broadcast(timestamp)
        self._timestamp = time.strftime('%Y%m%d_%H%M%S',
                                        time.localtime(timestamp.item()))

        # https://github.com/pytorch/pytorch/issues/973
        # set resource limit
        if platform.system() != 'Windows':
            import resource
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            base_soft_limit = rlimit[0]
            hard_limit = rlimit[1]
            soft_limit = min(max(resource_limit, base_soft_limit), hard_limit)
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (soft_limit, hard_limit))

        self._randomness = randomness
        self._set_randomness(**randomness)

    def _setup_distributed(self, *args, **kwargs):
        """Setup distributed environment."""
        pass

    def _set_randomness(
        self,
        seed: Optional[int] = None,
        diff_rank_seed: bool = False,
        deterministic: bool = False,
    ) -> None:
        """Set random seed to guarantee reproducible results.

        Args:
            seed (int, optional): A number to set random modules.
                Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds according
                to global rank. Defaults to False.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Defaults to False.
                See https://pytorch.org/docs/stable/notes/randomness.html for
                more details.
        """
        from mmengine.runner import set_random_seed
        self._seed = set_random_seed(
            seed=seed,
            deterministic=deterministic,
            diff_rank_seed=diff_rank_seed)

    def build_model(self, model: Union[nn.Module, dict]) -> nn.Module:
        """Build model.

        If ``model`` is a dict, it will be used to build a ``nn.Module``
        object. Otherwise, if ``model`` is a ``nn.Module`` object it will be
        returned directly.

        An example of ``model``::

            model = dict(type='ResNet')

        Args:
            model (nn.Module or dict): A ``nn.Module`` object or a dict to
                build ``nn.Module`` object. If ``model`` is a ``nn.Module``
                object, just returns itself.

        Note:
            The returned model must implement ``train_step``, ``test_step``
            if ``runner.train`` or ``runner.test`` will be called. If
            ``runner.val`` will be called or ``val_cfg`` is configured,
            model must implement `val_step`.

        Returns:
            nn.Module: Model build from ``model``.
        """
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, dict):
            model = MODELS.build(model)
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')

    def compile_model(
        self,
        model: nn.Module,
        compile: Union[dict, bool] = False,
    ) -> nn.Module:
        """Compile model.

        Args:
            model (nn.Module): Model to compile.

        Returns:
            nn.Module: Compiled model.
        """
        if isinstance(compile, bool) and not compile:
            return model

        assert digit_version(TORCH_VERSION) >= digit_version('2.0.0'), (
            'PyTorch >= 2.0.0 is required to enable torch.compile')

        if isinstance(compile, bool):
            compile = dict()

        target = compile.pop('target', 'forward')
        func = getattr(model, target)
        compiled_func = torch.compile(func, **compile)
        setattr(model, target, compiled_func)
        self.logger.info('Model has been "compiled". The first few iterations '
                         'will be slow, please be patient.')

        return model

    def _init_model_weights(self, model: nn.Module) -> nn.Module:
        """Initialize the model weights if the model has
        :meth:`init_weights`"""
        if (hasattr(model, 'init_weights') and self.dispatch_kwargs.get(
                'init_weights_for_test_or_val', True)):
            model.init_weights()
            # sync params and buffers
            for _, params in model.state_dict().items():
                broadcast(params)

        return model

    def build_optim_wrapper(
        self,
        optim_wrapper: Union[Optimizer, BaseOptimWrapper, dict],
        model: Optional[nn.Module] = None,
    ) -> BaseOptimWrapper:
        """Build optimizer wrapper.

        If ``optim_wrapper`` is a config dict for only one optimizer,
        the keys must contain ``optimizer``, and ``type`` is optional.
        It will build a :obj:`OptimWrapper` by default.

        If ``optim_wrapper`` is a config dict for multiple optimizers, i.e.,
        it has multiple keys and each key is for an optimizer wrapper. The
        constructor must be specified since
        :obj:`DefaultOptimizerConstructor` cannot handle the building of
        training with multiple optimizers.

        If ``optim_wrapper`` is a dict of pre-built optimizer wrappers, i.e.,
        each value of ``optim_wrapper`` represents an ``OptimWrapper``
        instance. ``build_optim_wrapper`` will directly build the
        :obj:`OptimWrapperDict` instance from ``optim_wrapper``.

        Args:
            optim_wrapper (BaseOptimWrapper or dict): An OptimWrapper object or a
                dict to build OptimWrapper objects. If ``optim_wrapper`` is an
                OptimWrapper, just return an ``OptimizeWrapper`` instance.

        Note:
            For single optimizer training, if `optim_wrapper` is a config
            dict, `type` is optional(defaults to :obj:`OptimWrapper`) and it
            must contain `optimizer` to build the corresponding optimizer.

        Examples:
            >>> # build an optimizer
            >>> optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=dict(
            ...     type='SGD', lr=0.01))
            >>> # optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> # is also valid.
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build optimizer without `type`
            >>> optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build multiple optimizers
            >>> optim_wrapper_cfg = dict(
            ...    generator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='SGD', lr=0.01)),
            ...    discriminator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='Adam', lr=0.001))
            ...    # need to customize a multiple optimizer constructor
            ...    constructor='CustomMultiOptimizerConstructor',
            ...)
            >>> optim_wrapper = runner.optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            name: generator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.1
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            name: discriminator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            'discriminator': Adam (
            Parameter Group 0
                dampening: 0
                lr: 0.02
                momentum: 0
                nesterov: False
                weight_decay: 0
            )

        Important:
            If you need to build multiple optimizers, you should implement a
            MultiOptimWrapperConstructor which gets parameters passed to
            corresponding optimizers and compose the ``OptimWrapperDict``.
            More details about how to customize OptimizerConstructor can be
            found at `optimizer-docs`_.

        Returns:
            BaseOptimWrapper: Optimizer wrapper build from ``optimizer_cfg``.
        """  # noqa: E501
        if isinstance(optim_wrapper, BaseOptimWrapper):
            return optim_wrapper
        if isinstance(optim_wrapper, (dict, ConfigDict, Config)):
            # optimizer must be defined for single optimizer training.
            optimizer = optim_wrapper.get('optimizer', None)

            # If optimizer is a built `Optimizer` instance, the optimizer
            # wrapper should be built by `OPTIM_WRAPPERS` registry.
            if isinstance(optimizer, Optimizer):
                optim_wrapper.setdefault('type', 'OptimWrapper')
                return OPTIM_WRAPPERS.build(optim_wrapper)  # type: ignore

            # If `optimizer` is not None or `constructor` is defined, it means,
            # optimizer wrapper will be built by optimizer wrapper
            # constructor. Therefore, `build_optim_wrapper` should be called.
            if optimizer is not None or 'constructor' in optim_wrapper:
                assert model is not None
                return build_optim_wrapper(model, optim_wrapper)
            else:
                # if `optimizer` is not defined, it should be the case of
                # training with multiple optimizers. If `constructor` is not
                # defined either, each value of `optim_wrapper` must be an
                # `OptimWrapper` instance since `DefaultOptimizerConstructor`
                # will not handle the case of training with multiple
                # optimizers. `build_optim_wrapper` will directly build the
                # `OptimWrapperDict` instance from `optim_wrapper.`
                optim_wrappers = OrderedDict()
                for name, optim in optim_wrapper.items():
                    if not isinstance(optim, BaseOptimWrapper):
                        raise ValueError(
                            'each item mush be an optimizer object when '
                            '"type" and "constructor" are not in '
                            f'optimizer, but got {name}={optim}')
                    optim_wrappers[name] = optim
                return OptimWrapperDict(**optim_wrappers)
        else:
            raise TypeError('optimizer wrapper should be an OptimWrapper '
                            f'object or dict, but got {optim_wrapper}')

    def _build_param_scheduler(
        self,
        scheduler: Union[_ParamScheduler, Dict, List],
        optim_wrapper: BaseOptimWrapper,
        default_args: dict,
    ) -> List[_ParamScheduler]:
        """Build parameter schedulers for a single optimizer.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.
            optim_wrapper (BaseOptimWrapper): An optimizer wrapper object is
                passed to construct ParamScheduler object.

        Returns:
            list[_ParamScheduler]: List of parameter schedulers build from
            ``scheduler``.

        Note:
            If the train loop is built, when building parameter schedulers,
            it supports setting the max epochs/iters as the default ``end``
            of schedulers, and supports converting epoch-based schedulers
            to iter-based according to the ``convert_to_iter_based`` key.
        """
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        max_epochs = default_args.pop('max_epochs', None)
        max_iters = default_args.pop('max_iters', None)

        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)

                # Set default end
                if _scheduler.get('by_epoch', True):
                    if max_epochs is None:
                        raise ValueError(
                            'max_epochs must be specified in default_args')
                    default_end = max_epochs
                else:
                    if max_iters is None:
                        raise ValueError(
                            'max_iters must be specified in default_args')
                    default_end = max_iters
                _scheduler.setdefault('end', default_end)
                self.logger.debug(
                    f'The `end` of {_scheduler["type"]} is not set. '
                    'Use the max epochs/iters of train loop as default.')

                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optim_wrapper, **default_args)))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')
        return param_schedulers

    def build_param_scheduler(
        self,
        scheduler: Union[_ParamScheduler, Dict, List],
        optim_wrapper: BaseOptimWrapper,
        default_args: Optional[dict] = None,
    ) -> ParamSchedulerType:
        """Build parameter schedulers.

        ``build_param_scheduler`` should be called after
        ``build_optim_wrapper`` because the building logic will change
        according to the number of optimizers built by the runner.
        The cases are as below:

        - Single optimizer: When only one optimizer is built and used in the
          runner, ``build_param_scheduler`` will return a list of
          parameter schedulers.
        - Multiple optimizers: When two or more optimizers are built and used
          in runner, ``build_param_scheduler`` will return a dict containing
          the same keys with multiple optimizers and each value is a list of
          parameter schedulers. Note that, if you want different optimizers to
          use different parameter schedulers to update optimizer's
          hyper-parameters, the input parameter ``scheduler`` also needs to be
          a dict and its key are consistent with multiple optimizers.
          Otherwise, the same parameter schedulers will be used to update
          optimizer's hyper-parameters.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.

        Examples:
            >>> # build one scheduler
            >>> optim_cfg = dict(dict(type='SGD', lr=0.01))
            >>> runner.optim_wrapper = runner.build_optim_wrapper(
            >>>     optim_cfg)
            >>> scheduler_cfg = dict(type='MultiStepLR', milestones=[1, 2])
            >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
            >>> schedulers
            [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f6966290>]  # noqa: E501

            >>> # build multiple schedulers
            >>> scheduler_cfg = [
            ...    dict(type='MultiStepLR', milestones=[1, 2]),
            ...    dict(type='StepLR', step_size=1)
            ... ]
            >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
            >>> schedulers
            [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f60dd3d0>,  # noqa: E501
            <mmengine.optim.scheduler.lr_scheduler.StepLR at 0x7f70f6eb6150>]

        Above examples only provide the case of one optimizer and one scheduler
        or multiple schedulers. If you want to know how to set parameter
        scheduler when using multiple optimizers, you can find more examples
        `optimizer-docs`_.

        Returns:
            list[_ParamScheduler] or dict[str, list[_ParamScheduler]]: List of
            parameter schedulers or a dictionary contains list of parameter
            schedulers build from ``scheduler``.

        .. _optimizer-docs:
           https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html
        """
        if default_args is None:
            default_args = {}
            if 'epoch_length' in self.dispatch_kwargs:
                default_args['epoch_length'] = self.dispatch_kwargs[
                    'epoch_length']
            if 'max_epochs' in self.dispatch_kwargs:
                default_args['max_epochs'] = self.dispatch_kwargs['max_epochs']
            if 'max_iters' in self.dispatch_kwargs:
                default_args['max_iters'] = self.dispatch_kwargs['max_iters']

        param_schedulers: ParamSchedulerType
        if not isinstance(optim_wrapper, OptimWrapperDict):
            # Since `OptimWrapperDict` inherits from `OptimWrapper`,
            # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
            # whether `self.optim_wrapper` is an `OptimizerWrapper` or
            # `OptimWrapperDict` instance. Therefore, here we simply check
            # self.optim_wrapper is not an `OptimWrapperDict` instance and
            # then assert it is an OptimWrapper instance.
            assert isinstance(optim_wrapper, BaseOptimWrapper), (
                '`build_optimizer` should be called before'
                '`build_param_scheduler` because the latter depends '
                'on the former')
            param_schedulers = self._build_param_scheduler(
                scheduler, optim_wrapper, default_args)  # type: ignore
            return param_schedulers
        else:
            param_schedulers = dict()
            for name, optimizer in optim_wrapper.items():
                if isinstance(scheduler, dict) and 'type' not in scheduler:
                    # scheduler is a dict and each item is a ParamScheduler
                    # object or a config to build ParamScheduler objects
                    param_schedulers[name] = self._build_param_scheduler(
                        scheduler[name], optimizer, default_args)
                else:
                    param_schedulers[name] = self._build_param_scheduler(
                        scheduler, optimizer, default_args)

            return param_schedulers

    def _scale_lr(self) -> None:
        """Automatically scaling learning rate in training according to the
        ratio of ``base_batch_size`` in ``autoscalelr_cfg`` and real batch
        size.

        It scales the learning rate linearly according to the
        `paper <https://arxiv.org/abs/1706.02677>`_.

        Note:
            ``scale_lr`` must be called after building optimizer wrappers
            and before building parameter schedulers.
        """
        if (self._auto_scale_lr is None
                or not self._auto_scale_lr.get('enable', False)):
            return None

        assert 'base_batch_size' in self._auto_scale_lr, \
            'Lack of `base_batch_size` in `auto_scale_lr`.'

        try:
            real_bs = self.world_size * self.dispatch_kwargs['train_micro_batch_size_per_gpu']
        except:
            real_bs = self.world_size * self.train_micro_batch_size_per_gpu ## for FSDP strategy

        base_bs = self._auto_scale_lr['base_batch_size']
        ratio = float(real_bs) / float(base_bs)
        self.logger.info(f'LR is set based on batch size of {base_bs} '
                         f'and the current batch size is {real_bs}. '
                         f'Scaling the original LR by {ratio}.')

        def _is_built(schedulers):
            if isinstance(schedulers, dict):
                return False if 'type' in schedulers else any(
                    _is_built(s) for s in schedulers.values())
            if isinstance(schedulers, list):
                return any(_is_built(s) for s in schedulers)
            return isinstance(schedulers, _ParamScheduler)

        if _is_built(self.param_schedulers):
            raise RuntimeError('`scale_lr` should be called before building '
                               'ParamScheduler because ParamScheduler will '
                               'store initial lr from optimizer wrappers')

        assert isinstance(self.optim_wrapper, BaseOptimWrapper), \
            '`scale_lr should be called after building OptimWrapper'

        if isinstance(self.optim_wrapper, OptimWrapperDict):
            wrappers = list(self.optim_wrapper.values())
        else:
            wrappers = [self.optim_wrapper]  # type: ignore

        for wrapper in wrappers:
            for group in wrapper.optimizer.param_groups:
                group['lr'] = group['lr'] * ratio

    def build_logger(
        self,
        log_level: Union[int, str] = 'INFO',
        log_file: Optional[str] = None,
        **kwargs,
    ) -> MMLogger:
        """Build a global asscessable MMLogger.

        Args:
            log_level (int or str): The log level of MMLogger handlers.
                Defaults to 'INFO'.
            log_file (str, optional): Path of filename to save log.
                Defaults to None.
            **kwargs: Remaining parameters passed to ``MMLogger``.

        Returns:
            MMLogger: A MMLogger object build from ``logger``.
        """
        if log_file is None:
            log_file = osp.join(self.log_dir, f'{self._timestamp}.log')

        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', self.experiment_name)
        # `torch.compile` in PyTorch 2.0 could close all user defined handlers
        # unexpectedly. Using file mode 'a' can help prevent abnormal
        # termination of the FileHandler and ensure that the log file could
        # be continuously updated during the lifespan of the runner.
        log_cfg.setdefault('file_mode', 'a')

        return MMLogger.get_instance(**log_cfg)  # type: ignore

    def model_state_dict(self) -> dict:
        """Get model state dict."""
        from mmengine.runner import weights_to_cpu
        return weights_to_cpu(self.model.state_dict())

    def optim_state_dict(self) -> dict:
        """Get optimizer state dict."""
        if isinstance(self.optim_wrapper, BaseOptimWrapper):
            return self.optim_wrapper.state_dict()
        else:
            raise TypeError('self.optim_wrapper should be a `BaseOptimWrapper`'
                            f' instance, but got {self.optim_wrapper}')

    def scheduler_state_dict(self) -> Union[dict, list]:
        """Get parameter scheduler state dict."""
        if isinstance(self.param_schedulers, dict):
            state_dict: dict = dict()
            for name, schedulers in self.param_schedulers.items():
                state_dict[name] = []
                for scheduler in schedulers:
                    state_dict[name].append(scheduler.state_dict())
            return state_dict
        else:
            state_list = []
            for scheduler in self.param_schedulers:  # type: ignore
                state_list.append(scheduler.state_dict())
            return state_list

    def load_model_state_dict(
        self,
        state_dict: dict,
        *,
        strict: bool = False,
        revise_keys: list = [(r'^module.', '')],
    ) -> None:
        """Load model state from dict."""
        from mmengine.runner.checkpoint import _load_checkpoint_to_model

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        _load_checkpoint_to_model(model, state_dict, strict, revise_keys)

    def load_optim_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from dict."""
        self.optim_wrapper.load_state_dict(state_dict)

    def load_scheduler_state_dict(self, state_dict: Union[dict, list]) -> None:
        """Load scheduler state from dict."""
        if isinstance(self.param_schedulers, dict):
            assert isinstance(state_dict, dict)
            for name, schedulers in self.param_schedulers.items():
                for scheduler, ckpt_scheduler in zip(schedulers,
                                                     state_dict[name]):
                    scheduler.load_state_dict(ckpt_scheduler)
        else:
            for scheduler, ckpt_scheduler in zip(
                    self.param_schedulers,  # type: ignore
                    state_dict):
                scheduler.load_state_dict(ckpt_scheduler)

    def load_or_resume(
        self,
        *,
        load_from: Optional[str] = None,
        resume: Union[bool, str] = False,
    ) -> Optional[dict]:
        """Load checkpoint or resume from checkpoint.

        Args:
            load_from (str, optional): The checkpoint file to load from.
                Defaults to None.
            resume (bool or str): Whether to resume training. Defaults to
                False. If ``resume`` is True and ``load_from`` is None,
                automatically to find latest checkpoint from ``work_dir``.
                If not found, resuming does nothing. If ``resume`` is a string,
                it will be treated as the checkpoint file to resume from.
        """
        from mmengine.runner import find_latest_checkpoint

        if not resume and load_from is None:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if isinstance(resume, str):
            resume_from = resume
        elif resume and load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self._work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif resume and load_from is not None:
            # resume from the specified checkpoint
            resume_from = load_from

        if resume_from is not None:
            return self.resume(resume_from)
        elif load_from is not None:
            return self.load_checkpoint(load_from)

        return None

    @abstractmethod
    def load_checkpoint(
        self,
        filename: str,
        *,
        map_location: Union[str, Callable] = 'cpu',
        strict: bool = False,
        revise_keys: list = [(r'^module.', '')],
        callback: Optional[Callable] = None,
    ) -> dict:
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.

        Keyword Args:
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
            callback (callable, callable): Callback function to modify the
                checkpoint after loading the checkpoint.
                Defaults to None.
        """

    @abstractmethod
    def resume(
        self,
        filename: str,
        *,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: Union[str, Callable] = 'default',
        callback: Optional[Callable] = None,
    ) -> dict:
        """Resume training from given ``filename``.

        Four types of states will be resumed.

        - model state
        - optimizer state
        - scheduler state
        - randomness state

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.

        Keyword Args:
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
            callback (callable, callable): Callback function to modify the
                checkpoint before saving the checkpoint.
                Defaults to None.
        """

    @abstractmethod
    def save_checkpoint(
        self,
        filename: str,
        *,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        extra_ckpt: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """Save checkpoint to given ``filename``.

        Args:
            filename (str): Filename to save checkpoint.

        Keyword Args:
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            extra_ckpt (dict, optional): Extra checkpoint to save.
                Defaults to None.
            callback (callable, callable): Callback function to modify the
                checkpoint before saving the checkpoint.
                Defaults to None.
        """

    def collect_env(self) -> Tuple[dict, dict]:
        """Collect the information of the running environments."""
        system_env = collect_env()
        runtime_env: OrderedDict = OrderedDict()
        runtime_env.update(self._env_kwargs)
        runtime_env.update(self.randomness)
        runtime_env['Distributed launcher'] = self.launcher
        runtime_env['Distributed training'] = self.distributed
        runtime_env['GPU number'] = self.world_size

        return system_env, runtime_env

    def _prepared_components(self):
        return_items = [self.model]
        if hasattr(self, 'optim_wrapper'):
            return_items.append(self.optim_wrapper)

        if hasattr(self, 'param_schedulers'):
            return_items.append(self.param_schedulers)

        return return_items[0] if len(return_items) == 1 else return_items
