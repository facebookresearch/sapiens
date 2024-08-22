# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import inspect
import os
import os.path as osp
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch.nn as nn
from torch.distributed.fsdp import (FullStateDictConfig,
                                    FullyShardedDataParallel,
                                    LocalStateDictConfig, StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, LocalOptimStateDictConfig, OptimStateDictConfig,
    StateDictConfig)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import mmengine
from legion.common.worker_pool import WorkerPool
from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
from mmengine.dist import get_rank, is_main_process
from mmengine.model import BaseDataPreprocessor, is_model_wrapper
from mmengine.optim import (AmpOptimWrapper, BaseOptimWrapper, OptimWrapper,
                            OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (MODEL_WRAPPERS, OPTIM_WRAPPERS,
                               PARAM_SCHEDULERS, STRATEGIES, Registry)
from mmengine.utils import get_git_hash, mkdir_or_exist
from .distributed import DDPStrategy
from .utils import MetaTensorContext

FSDP = FullyShardedDataParallel
FSDP_CONFIGS = Registry('fsdp configs')
FSDP_CONFIGS.register_module(module=FullOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=LocalOptimStateDictConfig)
FSDP_CONFIGS.register_module(module=FullStateDictConfig)
FSDP_CONFIGS.register_module(module=LocalStateDictConfig)

def _save_checkpoint(params):
    from mmengine.runner.checkpoint import save_checkpoint

    checkpoint, filename = params
    save_checkpoint(checkpoint, filename)

@STRATEGIES.register_module()
class FSDPStrategy(DDPStrategy):
    """Support training model with FullyShardedDataParallel (FSDP).

    Keyword Args:
        model_wrapper (dict, optional): Config dict for model wrapper. The
            default configuration is:

            Examples:
                >>> model_wrapper = dict(
                >>>    type='MMFullyShardedDataParallel',
                >>>    use_orig_params=True,
                >>> )

            See more configurable arguments in
            :class:`MMFullyShardedDataParallel`. Defaults to None
        skip_init_weights (bool, optional): Whether to skip initialization of
            weights. Defaults to False. This is useful when the parameters of
            the large model are loaded from a checkpoint, since skipping the
            initialization of weights can save a lot of time.
        state_dict_cfg (str or dict): Configuration for
            how to save and load the state dict of the model, optimizer, and
            scheduler.

            - "local": save and load the sharded state dict in all ranks.
            - "full": save and load the full state dict in rank 0.
            - `dict` object: save and load the state dict more flexibly. For
              example, you can first offload the state dict to the 'cpu' and
              then save it to the disk. This can help you to load the
              checkpoint in a non-gpu environment:

              Examples:
                >>> state_dict_cfg=dict(
                >>>     state_dict_type='FULL_STATE_DICT',
                >>>     state_dict_config=dict(type='FullStateDictConfig', offload_to_cpu=True),
                >>>     optim_state_dict_config=dict(type='FullOptimStateDictConfig', offload_to_cpu=True),

              See more configurable arguments for ``state_dict_cfg``,
              ``state_dict_config``, and ``optim_state_dict_config``in
              `FSDP official api documents`_
        kwargs (dict): Additional arguments passed to :class:`DDPStrategy`:

            - work_dir (str): The working directory to save checkpoints.
              The logs will be saved in the subdirectory of `work_dir` named
              :attr:`timestamp`. Defaults to 'work_dirs'.
            - experiment_name (str, optional): Name of current experiment. If
              not specified, timestamp will be used as :attr:`experiment_name`.
              Defaults to None.
            - env_kwargs (dict, optional): Environment config passed in
              :meth:`setup_env`. Defaults to None.
            - log_kwargs (dict, optional): Logger config passed in
              :meth:`build_logger`. Defaults to None.

    .. _FSDP official api documents: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.set_state_dict_type
    """  # noqa: E501

    def __init__(self,
                 *,
                 model_wrapper: Optional[dict] = None,
                 skip_init_weights=False,
                 state_dict_cfg: Union[str, dict] = 'local',
                 train_micro_batch_size_per_gpu: Optional[int] = None,
                 **kwargs):
        super().__init__(model_wrapper=model_wrapper, **kwargs)
        self._init_state_dict_cfg(state_dict_cfg)
        if not isinstance(skip_init_weights, bool):
            raise TypeError('skip_init_weights must be a boolean, but got '
                            f'{type(skip_init_weights)}')
        self.skip_init_weights = skip_init_weights
        self.train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu

    def _wrap_model(self, model: nn.Module) -> None:
        """Wrap the model to :obj:``MMFullyShardedDataParallel`` or other
        custom fully sharded data parallel module wrappers.

        Args:
            model (nn.Module): Model to be wrapped.

        Returns:
            FullyShardedDataParallel: ``MMFullyShardedDataParallel``
            or subclass of ``FullyShardedDataParallel``.
        """
        for module in model.modules():
            if isinstance(module, BaseDataPreprocessor):
                module.to(get_device())

        if is_model_wrapper(model):
            return

        if self.model_wrapper is None:
            self.model_wrapper = dict(type='MMFullyShardedDataParallel')

        default_args = dict(
            module=model,
            device_id=int(os.environ['LOCAL_RANK']),
            type='MMFullyShardedDataParallel')
        model = MODEL_WRAPPERS.build(
            self.model_wrapper, default_args=default_args)
        model.set_state_dict_type(model, self.state_dict_type,
                                  self.state_dict_config,
                                  self.optim_state_dict_config)
        return model

    def _is_full_state_dict(self):
        """Whether to save and load the full state_dict in rank 0."""
        return self.state_dict_type == StateDictType.FULL_STATE_DICT

    # This is lazy initialized so each replicas creates its own if it needs one.
    @functools.cached_property
    def worker_pool(self):
        worker_pool = WorkerPool(1, _save_checkpoint)
        worker_pool.start()
        return worker_pool

    def build_model(self, model: Union[nn.Module, dict]) -> nn.Module:
        """Build model.

        If skip_init_weights is True, the model will be built with an empty
        weights. It means that :meth:`load_checkpoint` must be called to fill
        the weights before training.

        Args:
            model (nn.Module or dict): A ``nn.Module`` object or a dict to
                build ``nn.Module`` object. If ``model`` is a ``nn.Module``
                object, just returns itself.

        Returns:
            nn.Module: Model build from ``model``.
        """
        if self.skip_init_weights:
            if isinstance(model, dict):
                # Accelerate initialization by skipping init weights
                with MetaTensorContext():
                    model = super().build_model(model)
                model.to_empty(device='cpu')
        else:
            model = super().build_model(model)

        # `id_to_name` will be used to convert the `optim_state_dict` of the
        # raw optimizer to the `optim_state_dict`
        # returned by `FSDP.optim_state_dict` in
        # `StateDictType.FULL_STATE_DICT` mode.
        self.id_to_name = dict()
        for name, param in model.named_parameters():
            self.id_to_name[id(param)] = name
        return model

    def save_checkpoint(self,
                        filename: str,
                        *,
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        extra_ckpt: Optional[dict] = None,
                        callback: Optional[Callable] = None) -> None:
        """Save checkpoint to given ``filename``.

        If ``state_dict_type`` is `full`, the checkpoint will only be saved in
        rank0. The structure of the saved checkpoint is the same as the one
        saved by ``DDPStrategy``

        If ``state_dict_type`` is `local`, each rank will save the sharded
        state dict to a directory, which means the saved structure will look
        like this:

        .. code-block:: bash

            ── epoch_0.pth
                ├── rank0.pth
                ├── rank1.pth
                ├── ...
                └── rank8.pth

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
        from mmengine.runner.checkpoint import save_checkpoint

        state_dict: dict = dict()
        state_dict['state_dict'] = self.model_state_dict()

        # save optimizer state dict
        if save_optimizer and hasattr(self, 'optim_wrapper'):
            state_dict['optimizer'] = self.optim_state_dict()

        # save param scheduler state dict
        if save_param_scheduler and hasattr(self, 'param_schedulers'):
            state_dict['param_schedulers'] = self.scheduler_state_dict()

        # save extra checkpoint passed by users
        if extra_ckpt is None:
            extra_ckpt = dict()
        if 'meta' not in extra_ckpt:
            extra_ckpt['meta'] = dict()

        extra_ckpt['meta'].update(
            seed=self.seed,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine=mmengine.__version__ + get_git_hash(),
        )
        state_dict.update(extra_ckpt)

        # users can do some modification before saving checkpoint
        if callback is not None:
            callback(state_dict)

        # In non-FULL_STATE_DICT model, FSDPStrategy will save checkpoint
        # of different ranks in different files.
        if not self._is_full_state_dict():
            rank = get_rank()
            mkdir_or_exist(filename)
            ckpt_name = f'rank{rank}.pth'
            filename = osp.join(filename, ckpt_name)
            # Don't use worker_pool due to use of ShardedTensor
            _save_checkpoint((state_dict, filename))

        if is_main_process():
            if self._is_full_state_dict():
                self.worker_pool.put((state_dict, filename))
            else:
                # Don't use worker_pool due to use of ShardedTensor
                _save_checkpoint((state_dict, filename))

    def model_state_dict(self) -> dict:
        """Get model state dict based on the ``state_dict_type``.

        If ``state_dict_type`` is `full`, the model state dict will be the
        same as the one of original unsharded model.

        If ``state_dict_type`` is ``local``, and ``use_orig_params`` is ``True``
        in ``model_wrapper``. The key of the state dict will be the same as
        the one of original unsharded model, but its value will be the sharded
        one

        If ``state_dict_type`` is `local`, and ```use_orig_params``` is
        ``False`` in ``model_wrapper``, the flatten and sharded state dict will
        be returned.

        See more details in the `official api documents`_

        .. _official api documents: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict
        """  # noqa: E501
        # We've set state_dict by `FSDP.set_state_dict_type`, therefore we
        # should get model state dict by `FSDP.state_dict`
        return self.model.state_dict()

    def optim_state_dict(self) -> dict:
        """Get model state dict based on the ``state_dict_type``.

        If ``state_dict_type`` is ``full``, the optimizer state dict can be
        loaded by the original unsharded optimizer.

        Otherwise, the optimizer state dict could only be loaded by the
        optimizer with sharded parameters.

        Note:
            The optimizer state dict is not the same as the one of original
            optimizer even if in ``full`` mode, although they can be loaded
            correctly.

        See more details in the `official api documents`_

        .. _official api documents: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.optim_state_dict
        """  # noqa: E501
        return FSDP.optim_state_dict(self.model, self.optim_wrapper)

    def load_checkpoint(self, filename: str, **kwargs) -> dict:
        """Load checkpoint from given ``filename``.

        Note:
            If ``state_dict_type`` is `local`, the filename should be a
            directory contains ``rank{i}.pth``.

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
        if self._is_full_state_dict():
            return super(DDPStrategy, self).load_checkpoint(filename, **kwargs)
        else:
            rank = get_rank()
            filename = osp.join(filename, f'rank{rank}.pth')
            return super(DDPStrategy, self).load_checkpoint(filename, **kwargs)

    def load_model_state_dict(
        self,
        state_dict: dict,
        *,
        strict: bool = False,
        revise_keys: list = [(r'^module.', '')],
    ) -> None:  # type: ignore
        """Load model state from dict.

        Warning:
            `revise_keys` is not supported yet.

        Args:
            state_dict (dict): Model state dict returned by
                :meth:`FSDPStrategy.model_state_dict`. If ``state_dict_type``
                is ``full``. ``state_dict`` could be the result of
                ``model.state_dict()``
            strict (bool): Whether to load model state dict strictly.
                Defaults to False.
        """
        # We should load state dict by `FSDP.load_state_dict`
        self.model.load_state_dict(state_dict, strict=strict)

    def load_optim_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from dict.

        Args:
            state_dict (dict): The optimizer state dict. If ``state_dict_type``
                is ``full``. ``state_dict`` could be the result of
                ``optimizer.state_dict()``
        """
        # optim_state_dict = FSDP.optim_state_dict_to_load(state_dict, self.model, self.optim_wrapper.optimizer) ## old fsdp
        
        ## correct order of args in latest pytorch
        # https://github.com/pytorch/pytorch/blob/f3df7deab8953af76ff1723ed49094208057a834/torch/distributed/fsdp/fully_sharded_data_parallel.py#L1847
        optim_state_dict = FSDP.optim_state_dict_to_load(
                        model=self.model,
                        optim=self.optim_wrapper.optimizer,
                        optim_state_dict=state_dict)

        self.optim_wrapper.load_state_dict(optim_state_dict)

    def _init_state_dict_cfg(self, state_dict_cfg: Union[str, dict]) -> None:
        """Make ``state_dict_type`` and ``state_dict_config`` can be configured
        with string."""
        if isinstance(state_dict_cfg, str):
            if state_dict_cfg == 'full':
                self.state_dict_type = StateDictType.FULL_STATE_DICT
                self.state_dict_config = FullStateDictConfig(
                    rank0_only=True, offload_to_cpu=True)
                self.optim_state_dict_config = FullOptimStateDictConfig(
                    rank0_only=True, offload_to_cpu=True)
            elif state_dict_cfg == 'local':
                self.state_dict_type = StateDictType.LOCAL_STATE_DICT
                self.state_dict_config = LocalStateDictConfig()
                self.optim_state_dict_config = LocalOptimStateDictConfig()
            else:
                raise ValueError('FSDP only supports `full` and `local` '
                                 f'state_dict_type, but got {state_dict_cfg}')
        elif isinstance(state_dict_cfg, dict):
            if 'state_dict_type' not in state_dict_cfg:
                self.state_dict_type = StateDictType.LOCAL_STATE_DICT
            else:
                state_dict_type = state_dict_cfg['state_dict_type']
                if isinstance(state_dict_type, str):
                    self.state_dict_type = StateDictType[
                        state_dict_cfg['state_dict_type']]
                else:
                    self.state_dict_type = state_dict_type
            state_dict_config = state_dict_cfg.get('state_dict_config')
            if state_dict_config is None:
                self.state_dict_config = LocalStateDictConfig()
            elif isinstance(state_dict_config, dict):
                self.state_dict_config = FSDP_CONFIGS.build(
                    state_dict_cfg['state_dict_config'])
            else:
                self.state_dict_config = state_dict_config

            optim_state_dict_config = state_dict_cfg.get(
                'optim_state_dict_config')
            if optim_state_dict_config is None:
                self.optim_state_dict_config = LocalOptimStateDictConfig()
            elif isinstance(optim_state_dict_config, dict):
                self.optim_state_dict_config = FSDP_CONFIGS.build(
                    state_dict_cfg['optim_state_dict_config'])
            else:
                self.optim_state_dict_config = optim_state_dict_config
        else:
            raise TypeError('state_dict_cfg should be a `str` or a `dict`, '
                            f'but got {type(state_dict_cfg)}')

        if not isinstance(self.state_dict_type, StateDictType):
            raise TypeError('state_dict_type must be StateDictType, but got '
                            f'{type(self.state_dict_type)}')
        if not isinstance(self.state_dict_config, StateDictConfig):
            raise TypeError('state_dict_config must be StateDictConfig, but '
                            f'got {type(self.state_dict_config)}')
        if not isinstance(self.optim_state_dict_config, OptimStateDictConfig):
            raise TypeError('optim_state_dict_config must be '
                            'OptimStateDictConfig, but got '
                            f'{type(self.optim_state_dict_config)}')

    def build_optim_wrapper(
        self,
        optim_wrapper: Union[Optimizer, OptimWrapper, dict],
        model: Optional[nn.Module] = None,
    ) -> BaseOptimWrapper:
        """Support sharding the optimizer state dict given a built optimizer or
        optim_wrapper.

        See specific usage in :meth:`BaseStrategy.build_optim_wrapper`.
        """
        if isinstance(optim_wrapper, Optimizer):
            optim_wrapper = OptimWrapper(optim_wrapper)
        if isinstance(optim_wrapper, BaseOptimWrapper):
            assert model is not None
            # NOTE: The only difference is that FSDPStrategy will shard
            # the the built OptimWrapper
            optimizer = optim_wrapper.optimizer
            param_groups = optimizer.param_groups
            optim_state_dict = optimizer.state_dict()
            assert not optim_state_dict['state'], (
                'Optimizer state_dict should be empty when giving an built '
                'optim_wrapper to FSDPStrategy')
            # Align the state_dict with state_dict generated by
            # FSDP.full_optim_state_dict
            new_param_groups = []
            for group in param_groups:
                new_group = {
                    key: value
                    for key, value in group.items() if key != 'param'
                }
                new_group['params'] = [
                    self.id_to_name[id(param)] for param in group['params']
                ]
                new_param_groups.append(new_group)
            optim_state_dict['param_groups'] = new_param_groups
            defaults = {
                k: v
                for k, v in optimizer.defaults.items() if k != 'differentiable'
            }

            params_dict = {}
            for k, v in model.named_parameters():
                if '_fsdp_wrapped_module' in k:
                    k = k.replace('_fsdp_wrapped_module.', '')
                params_dict[k] = v

            params = []
            for param_group in new_param_groups:
                _params = []
                for param_name in param_group['params']:
                    if param_name not in params_dict:
                        raise RuntimeError(
                            'Failed to reconstruct the sharded optimizer. '
                            'You can try to set `use_orig_params=True` in '
                            '`model_wrapper`')
                    _params.append(params_dict[param_name])
                param_group = {
                    k: v
                    for k, v in param_group.items() if k != 'param'
                }
                param_group['params'] = _params
                params.append(param_group)

            new_optimizer = optimizer.__class__(params, **defaults)

            # Force to load the converted optim_state_dict in full mode.
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                optim_state_dict = FSDP.optim_state_dict_to_load(
                    optim_state_dict, model, new_optimizer)
                new_optimizer.load_state_dict(optim_state_dict)
            optim_wrapper.optimizer = new_optimizer
            return optim_wrapper
        if isinstance(optim_wrapper, (dict, ConfigDict, Config)):
            assert model is not None
            # optimizer must be defined for single optimizer training.
            optimizer = optim_wrapper.get('optimizer', None)
            optim_wrapper.setdefault('type', 'OptimWrapper')
            if optim_wrapper.get('type',
                                 'AmpOptimWrapper') in ('AmpOptimWrapper',
                                                        AmpOptimWrapper):
                optim_wrapper.setdefault('use_fsdp', True)

            # If optimizer is a built `Optimizer` instance, the optimizer
            # wrapper should be built by `OPTIM_WRAPPERS` registry.
            if isinstance(optimizer, Optimizer):
                return OPTIM_WRAPPERS.build(optim_wrapper)  # type: ignore

            # If `optimizer` is not None or `constructor` is defined, it means,
            # optimizer wrapper will be built by optimizer wrapper
            # constructor. Therefore, `build_optim_wrapper` should be called.
            if optimizer is not None or 'constructor' in optim_wrapper:
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
                    if not isinstance(optim, OptimWrapper):
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
        """Override this method to update the scheduler with the reconstructed
        sharded optimzer."""
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        max_epochs = default_args.pop('max_epochs', None)
        max_iters = default_args.pop('max_iters', None)

        param_schedulers = []
        for scheduler in schedulers:
            # Update the built scheduler with the sharded optimizer
            if isinstance(scheduler, (_ParamScheduler, LRScheduler)):
                parameter_keys = inspect.signature(
                    scheduler.__class__).parameters.keys()
                kwargs = {
                    k: v
                    for k, v in scheduler.state_dict().items()
                    if k in parameter_keys
                }
                scheduler = scheduler.__class__(optim_wrapper, **kwargs)
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
