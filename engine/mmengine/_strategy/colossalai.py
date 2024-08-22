# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import colossalai
    import colossalai.booster.mixed_precision as colo_precision
    import colossalai.booster.plugin as colo_plugin
    import colossalai.nn.optimizer as colo_optimizer
    from colossalai.booster import Booster
    from colossalai.interface import ModelWrapper
except Exception as e:  # noqa: F841
    colossalai = None
    colo_precision = None
    colo_plugin = None
    colo_optimizer = None
    Booster = None
    ModelWrapper = None

import torch
import torch.nn as nn

import mmengine
from mmengine import mkdir_or_exist
from mmengine._strategy import BaseStrategy
from mmengine.device import get_device
from mmengine.dist import init_dist, is_main_process
from mmengine.fileio import join_path
from mmengine.model import BaseDataPreprocessor
from mmengine.optim import BaseOptimWrapper, OptimWrapper, _ParamScheduler
from mmengine.registry import STRATEGIES, Registry
from mmengine.registry.root import MODEL_WRAPPERS, OPTIM_WRAPPERS, OPTIMIZERS
from mmengine.runner.checkpoint import _load_checkpoint, save_checkpoint
from mmengine.utils import get_git_hash

# Component for colossalai `plugins` and `mixed_precisions`
PLUGINS = Registry('plugin')
MIXED_PRECISIONS = Registry('mixed_precision')


def register_plugins():
    _plugins = inspect.getmembers(
        colo_plugin,
        lambda x: inspect.isclass(x) and issubclass(x, colo_plugin.Plugin))

    for name, plugin in _plugins:
        PLUGINS.register_module(name=name, module=plugin)


def register_optimizers():
    _colo_optimizer = inspect.getmembers(
        colo_optimizer,
        lambda x: inspect.isclass(x) and issubclass(x, torch.optim.Optimizer))
    for name, optim_type in _colo_optimizer:
        OPTIMIZERS.register_module(name=name, module=optim_type, force=True)


def register_mixed_precisions():
    _mixed_precisions = inspect.getmembers(
        colo_precision, lambda x: inspect.isclass(x) and issubclass(
            x, colo_precision.MixedPrecision))

    for name, mixed_precision in _mixed_precisions:
        MIXED_PRECISIONS.register_module(name=name, module=mixed_precision)


@OPTIM_WRAPPERS.register_module()
class ColossalAIOptimWrapper(OptimWrapper):
    """OptimWrapper for ColossalAI.

    The available optimizers are:
      - CPUAdam
      - FusedAdam
      - FusedLAMB
      - FusedSGD
      - HybridAdam
      - Lamb
      - Lars

    You can find more details in the `colossalai tutorial`_

    Args:
        optimizer (dict or torch.optim.Optimizer): The optimizer to be
            wrapped.
        accumulative_counts (int): The number of iterations to accumulate
            gradients. The parameters will be updated per
            ``accumulative_counts``.

    .. _colossalai tutorial: https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/nn/optimizer
    """  # noqa: E501

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 booster: Booster,
                 accumulative_counts: int = 1):
        super().__init__(optimizer, accumulative_counts=accumulative_counts)
        self.booster = booster

    @contextmanager
    def optim_context(self, model: nn.Module):
        if self.booster.plugin.support_no_sync():
            sync_context = self.booster.no_sync(model, self.optimizer)
        else:
            yield
            return
        if not self.should_sync():
            with sync_context:
                yield

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        self._inner_count += 1
        self.optimizer.backward(loss, **kwargs)


@MODEL_WRAPPERS.register_module()
class CollosalAIModelWrapper:

    def __init__(self, model_wrapper: ModelWrapper, model: nn.Module):
        self.model_wrapper = model_wrapper
        self.model = model

    def __call__(self, *args, **kwargs) -> Any:
        return self.model_wrapper(*args, **kwargs)

    def train_step(
        self,
        data: Union[dict, tuple, list],
        optim_wrapper: ColossalAIOptimWrapper,
    ) -> Dict[str, torch.Tensor]:
        data = self.model.data_preprocessor(data, training=True)
        with optim_wrapper.optim_context(self.model):
            losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.model.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.model.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')

    test_step = val_step

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self.model_wrapper(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self.model_wrapper(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def __getattr__(self, name):
        if hasattr(self.model_wrapper, name):
            return getattr(self.model_wrapper, name)
        elif hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            raise AttributeError(
                f'{self.model_wrapper} and {self.model} has no '
                f'attribute {name}')


@STRATEGIES.register_module()
class ColossalAIStrategy(BaseStrategy):
    """
    Args:
        config: (str or dict): The colossalai config file to setup distributed
            environment. See more details in the `colossalai config tutorial`_.
        mixed_precision (str or MixedPrecision): The mixed precision to run the
            training. Defaults to None. If the argument is a string, it can be
            'fp16', 'fp16_apex', 'bf16', or 'fp8' fp16' would use PyTorch AMP
            while `fp16_apex` would use Nvidia Apex.
        plugin (Plugin): The plugin to run the training. The type of `plugin`
            could be:

            - str: The available plugins are ``gemini`` and ``lowlevel-zero``.

              ``gemini`` means a `ZeRO`_ implementation with chunk-based
              memory management. You could find more details in the
              `colossalai gemini tutorial`_. ``lowlevel-zero`` means a
              Zero-1 and Zero-2 implementation. Although gemini is more
              memory saving, some unexpceted error could happen for
              some spectial model structure. lowlevel-zero is more stable.

            - dict: **dict-type style config to build a colossalai plugin**.

              See the `booster plugin tutorial`_ for more details.

        model_wrapper (dict, optional): Dict for model wrapper. Defaults to
            None.
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

    .. _colossalai config tutorial: https://colossalai.org/docs/basics/configure_parallelization
    .. _ZeRO: https://arxiv.org/abs/1910.02054
    .. _colossalai gemini tutorial: https://colossalai.org/docs/features/zero_with_chunk/#geminiddp
    .. _booster plugin tutorial: https://colossalai.org/docs/basics/booster_plugins

    """  # noqa: E501
    OPTIMIZER_DIR = 'optimizer'  # directory to save optimizer state.
    MODEL_DIR = 'model'  # directory to save model
    SCHEDULER_DIR = 'scheduler'  # directory to save scheduelrs
    model: CollosalAIModelWrapper  # type: ignore
    optim_wrapper: ColossalAIOptimWrapper  # type: ignore

    def __init__(
        self,
        *,
        config: Union[str, dict, None] = None,
        mixed_precision: Union[str, dict, None] = None,
        plugin: str = 'gemini',
        model_wrapper: Optional[dict] = None,
        **kwargs,
    ):
        if colossalai is None:
            raise ModuleNotFoundError(
                'Please install colossalai by `pip install -U colossalai`')
        register_plugins()
        register_mixed_precisions()
        register_optimizers()

        self.config = config or {}
        super().__init__(**kwargs)
        if mixed_precision is not None:
            mixed_precision = self._build_mixed_precision(mixed_precision)

        if plugin is not None:
            plugin = self._build_plugin(plugin)
        self.booster = Booster(mixed_precision=mixed_precision, plugin=plugin)
        self.model_wrapper = model_wrapper

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
                can be a dict used for build a model.

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
                If ``accumulative_counts`` is set in ``optim_wrapper``, you
                need to provide ``max_iters`` in ``dispatch_kwargs``.
        """
        if self._prepared:
            return self._prepared_components()
        if dispatch_kwargs is not None:
            self.dispatch_kwargs.update(dispatch_kwargs)

        model = self.build_model(model)
        model = self._init_model_weights(model)

        # optim_wrapper is required by booster
        if optim_wrapper is not None and isinstance(optim_wrapper, dict):
            optim_wrapper.setdefault('type', 'ColossalAIOptimWrapper')
            optim_wrapper.setdefault('booster', self.booster)
            optim_wrapper_type = OPTIM_WRAPPERS.get(optim_wrapper['type'])
            if optim_wrapper_type is None:
                raise ValueError(f'Failed to find {optim_wrapper["type"]} in '
                                 '`OPTIM_WRAPPERS`.')
            if 'clip_grad' in optim_wrapper:
                raise ValueError('`Please configure `clip_grad` in `plugin`')
            if not issubclass(optim_wrapper_type, ColossalAIOptimWrapper):
                raise ValueError(
                    'The type of `optim_wrapper` must be '
                    '`ColossalAIOptimWrapper` (or subclass), but got '
                    f'{optim_wrapper_type}')
            optim_wrapper = self.build_optim_wrapper(optim_wrapper, model)

        if optim_wrapper is not None:
            self.model, self.optim_wrapper = self._wrap(
                model, optim_wrapper)  # type: ignore
        else:
            self.model = self._wrap(model)  # type: ignore
        # TODO: Check whether `compile` is compatible with colossalai.

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, optim_wrapper)  # type: ignore

        if optim_wrapper is not None:
            self._scale_lr()
            accumulative_counts = getattr(self.optim_wrapper,
                                          '_accumulative_counts', 1)
            if accumulative_counts > 1:
                if 'max_iters' not in self.dispatch_kwargs:
                    raise ValueError(
                        '"max_iters" must be specified because '
                        '"accumulative_counts" was set as '
                        f'{accumulative_counts} which is greater than 1.')

                self.optim_wrapper.initialize_count_status(  # type: ignore
                    self.model, 0, self.dispatch_kwargs['max_iters'])
        self._prepared = True
        return self._prepared_components()

    def resume(
        self,
        filename: str,
        *,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: Union[str, Callable] = 'default',
        callback: Optional[Callable] = None,
    ) -> dict:
        """override this method since colossalai resume optimizer from filename
        directly."""
        self.logger.info(f'Resume checkpoint from {filename}')

        extra_ckpt = self.load_checkpoint(
            filename, map_location=map_location, callback=callback)

        if resume_optimizer:
            self.booster.load_optimizer(
                self.optim_wrapper.optimizer,
                join_path(filename, self.OPTIMIZER_DIR))

        if resume_param_scheduler:
            schedulers_dir = join_path(filename, self.SCHEDULER_DIR)
            for i, scheduler in enumerate(self.param_schedulers):
                self.booster.load_lr_scheduler(
                    scheduler, f'{schedulers_dir}/scheduler_{i}.pth')

        # resume random seed
        resumed_seed = extra_ckpt['meta'].get('seed', None)
        current_seed = self._randomness.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                self.logger.warning(f'The value of random seed in the '
                                    f'checkpoint "{resumed_seed}" is '
                                    f'different from the value in '
                                    f'`randomness` config "{current_seed}"')
            self._randomness.update(seed=resumed_seed)
            self._set_randomness(**self._randomness)

        # resume iter
        self.dispatch_kwargs['cur_iter'] = extra_ckpt['meta']['iter']

        return extra_ckpt

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

        Warning:
            `map_localtion` and `callback` parameters are not supported yet.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
        """
        self.logger.info(f'Load checkpoint from {filename}')
        self.booster.load_model(self.model.model_wrapper,
                                join_path(filename, self.MODEL_DIR))
        meta = _load_checkpoint(osp.join(filename, 'meta.pth'))
        return meta

    def save_checkpoint(
        self,
        filename: str,
        *,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        extra_ckpt: Optional[dict] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        # The checkpoint directory will be:
        # |--epoch_0.pth
        #    |---model/
        #    |---optimizer/
        #    |---scheduler/
        if extra_ckpt is None:
            extra_ckpt = dict()
        if 'meta' not in extra_ckpt:
            extra_ckpt['meta'] = dict()
        extra_ckpt['meta'].update(
            seed=self.seed,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine=mmengine.__version__ + get_git_hash())

        model_dir = join_path(filename, self.MODEL_DIR)
        optimizer_dir = join_path(filename, self.OPTIMIZER_DIR)
        schedulers_dir = join_path(filename, self.SCHEDULER_DIR)
        mkdir_or_exist(model_dir)
        mkdir_or_exist(optimizer_dir)
        mkdir_or_exist(schedulers_dir)

        self.booster.save_model(
            self.model.model_wrapper, checkpoint=model_dir, shard=True)

        if save_optimizer:
            self.booster.save_optimizer(
                self.optim_wrapper.optimizer,
                checkpoint=optimizer_dir,
                shard=True)

        if is_main_process() and save_param_scheduler:
            for i, scheduler in enumerate(self.param_schedulers):
                self.booster.save_lr_scheduler(
                    scheduler, f'{schedulers_dir}/scheduler_{i}.pth')

        save_checkpoint(extra_ckpt, join_path(filename, 'meta.pth'))

    def _build_plugin(self, plugin: Union[str, dict]):
        if isinstance(plugin, str):
            if plugin == 'gemini':
                plugin = colo_plugin.GeminiPlugin(
                    precision='bf16', placement_policy='cuda')
            elif plugin == 'lowlevel-zero':
                plugin = colo_plugin.LowLevelZeroPlugin()
            else:
                raise ValueError('`plugin` must be "gemini" or '
                                 '"lowlevel-zero"')
        elif isinstance(plugin, dict):
            plugin = PLUGINS.build(plugin)
        else:
            raise ValueError('`plugin` must be dict or str, but got a '
                             f'{type(plugin)} object)')
        return plugin

    def _build_mixed_precision(self, mixed_precision: Union[str, dict]):
        if isinstance(mixed_precision, str):
            if mixed_precision == 'fp16':
                mixed_precision = colo_precision.FP16TorchMixedPrecision()
            elif mixed_precision == 'fp16_apex':
                mixed_precision = colo_precision.FP16ApexMixedPrecision()
            elif mixed_precision == 'bf16':
                mixed_precision = colo_precision.BF16MixedPrecision()
            elif mixed_precision == 'fp8':
                mixed_precision = colo_precision.FP8MixedPrecision()
            else:
                raise ValueError(
                    'If `mixed_precision` is a string, it must be one of '
                    '"fp16", "fp16_apex", "bf16" and "fp8", but got '
                    f'{mixed_precision}')
        elif isinstance(mixed_precision, dict):
            mixed_precision = MIXED_PRECISIONS.build(mixed_precision)
        else:
            raise ValueError('mixed precision should be dict or str, but got '
                             f'a {type(mixed_precision)} object')
        return mixed_precision

    def _wrap(
        self,
        model: nn.Module,
        optim_wrapper: Optional[OptimWrapper] = None,
    ) -> Union[Tuple[CollosalAIModelWrapper, ColossalAIOptimWrapper],
               CollosalAIModelWrapper]:  # type: ignore
        """Wrap model with :class:`ModelWrapper`."""
        if self.model_wrapper is None:
            self.model_wrapper = {'type': 'CollosalAIModelWrapper'}

        # For zero series parallel, move `data_preprocessor` to current device
        # is reasonable. We need to `BaseDataPreprocessor.to` manually since
        # framework like colossalai and deepspeed could not handle it, leading
        # to `data_preprocessor` move data to cpu.
        for module in model.modules():
            if isinstance(module, BaseDataPreprocessor):
                module.to(get_device())

        if optim_wrapper is not None:
            optimizer = optim_wrapper.optimizer
            if not hasattr(optimizer, '_hook_for_profile'):
                # PyTorch 2.0 removes the `_hook_for_profile` in
                # `torch.optim.Optimizer`. We maintain this function here to
                # keep compatibility.
                # TODO: Remove this hardcode when ColossalAI supports
                # PyTorch 2.0
                optimizer.__class__._hook_for_profile = object

            # We do not pass `scheduler` and `Dataloader` here for:
            # 1. `Booster.boost` cannot accept a list of schedulers.
            # 2. `Strategy` cannot not accept dataloader now.
            model_wrapper, optimizer, *_ = self.booster.boost(model, optimizer)
            optim_wrapper.optimizer = optimizer
            default_args = {'model_wrapper': model_wrapper, 'model': model}
            model_wrapper = MODEL_WRAPPERS.build(
                self.model_wrapper, default_args=default_args)
            return model_wrapper, optim_wrapper  # type: ignore
        else:
            model_wrapper, *_ = self.booster.boost(model)
            default_args = {'model_wrapper': model_wrapper, 'model': model}
            model_wrapper = MODEL_WRAPPERS.build(
                self.model_wrapper, default_args=default_args)
            return model_wrapper

    def _setup_distributed(  # type: ignore
        self,
        launcher: Optional[str] = None,
        backend: str = 'nccl',
        **kwargs,
    ):
        init_dist(
            launcher, backend, init_backend='colossalai', config=self.config)
