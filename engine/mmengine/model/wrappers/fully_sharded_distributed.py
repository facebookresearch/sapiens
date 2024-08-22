# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
# yapf: disable
from torch.distributed.fsdp.api import (FullStateDictConfig,
                                        LocalOptimStateDictConfig,
                                        LocalStateDictConfig,
                                        OptimStateDictConfig,
                                        ShardedOptimStateDictConfig,
                                        ShardedStateDictConfig,
                                        ShardingStrategy, StateDictConfig,
                                        StateDictSettings, StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch, CPUOffload, FullOptimStateDictConfig,
    FullyShardedDataParallel, MixedPrecision)

# yapf: enable
from mmengine.optim import OptimWrapper
from mmengine.registry import FUNCTIONS, MODEL_WRAPPERS
from mmengine.structures import BaseDataElement
from mmengine.utils import digit_version, is_seq_of


@MODEL_WRAPPERS.register_module()
class MMFullyShardedDataParallel(FullyShardedDataParallel):
    """A wrapper for sharding Module parameters across data parallel workers.

    Different from FullyShardedDataParallel, MMFullyShardedDataParallel
    implements three methods :meth:`train_step`, :meth:`val_step` and
    :meth:`test_step`, which will be called by ``train_loop``, ``val_loop``
    and ``test_loop``.

    - ``train_step``: Called by ``runner.train_loop``, and implement
      default model forward, gradient back propagation, parameter updating
      logic.

    - ``val_step``: Called by ``runner.val_loop`` and get the inference
      results. Specially, since MMFullyShardedDataParallel will wrap model
      recursively, it may cause some problem if one just use
      ``BaseModel.val_step`` to implement ``val_step`` here. To avoid that,
      ``val_step`` will call methods of :obj:`BaseModel` to pre-process
      data first, and use ``FullyShardedDataParallel.forward`` to get result.

    - ``test_step``: Called by ``runner.test_loop`` and get the inference
      results. Its logic is equivalent to ``val_loop``.

    Args:
        module (nn.Module): module to be wrapped with FSDP.
        process_group (ProcessGroup, optional): process group for sharding.
        cpu_offload (bool, CPUOffload, optional):
            CPU offloading config.
            Different from FullyShardedDataParallel,Since it can be set by
            users' pre-defined config in MMEngine,its type is expected to be
            `None`, `bool` or `CPUOffload`.

            Currently, only parameter and gradient CPU offload is supported.
            It can be enabled via passing in
            ``cpu_offload=CPUOffload(offload_params=True)``. Note that this
            currently implicitly enables gradient offloading to CPU in order
            for params and grads to be on same device to work with optimizer.
            This API is subject to change. Default is ``None`` in which case
            there will be no offloading.
        auto_wrap_policy (str or Callable, optional):
            Specifying a policy to recursively wrap layers with FSDP.
            Different from FullyShardedDataParallel, Since it can be set by
            users' pre-defined config in MMEngine, its type is expected to be
            `None`, `str` or `Callable`. If it's `str`, then
            MMFullyShardedDataParallel will try to get specified method in
            ``FSDP_WRAP_POLICIES`` registry,and this method will be passed to
            FullyShardedDataParallel to finally initialize model.

            Note that this policy currently will only apply to child modules of
            the passed in module. The remainder modules are always wrapped in
            the returned FSDP root instance.
            ``default_auto_wrap_policy`` written in
            ``torch.distributed.fsdp.wrap`` is an example of
            ``auto_wrap_policy`` callable, this policy wraps layers with
            parameter sizes larger than 100M. Users can supply the customized
            ``auto_wrap_policy`` callable that should accept following
            arguments: ``module: nn.Module``, ``recurse: bool``,
            ``unwrapped_params: int``, extra customized arguments could be
            added to the customized ``auto_wrap_policy`` callable as well.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     unwrapped_params: int,
                >>>     # These are customizable for this policy function.
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return unwrapped_params >= min_num_params

        backward_prefetch (str or BackwardPrefetch, optional):
            Different from FullyShardedDataParallel, this argument could be a
            string or a BackwardPrefetch instance. If it's a string, then
            it should be ``BACKWARD_PRE`` or ``BACKWARD_POST``
        mixed_precision  (dict or MixedPrecision, optional):
            This configures native mixed precision for FSDP. If this is set to
            ``None``. Different from the native FSDP, this argument can a dict
            like this:

            Examples:
                >>> mixed_precision=dict(param_dtype='float16',
                >>>                      buffer_dtype='float32',
                >>>                      reduce_dtype='float32')

            Defaults to None.
        use_orig_params (bool): Different from native
            ``FullyShardedDataParallel``, it defaults to True.
        **kwargs: Keyword arguments passed to
            :class:`FullyShardedDataParallel`.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: Union[dict, ProcessGroup, None] = None,
        sharding_strategy: Union[str, ShardingStrategy] = None,
        cpu_offload: Union[bool, CPUOffload, None] = None,
        auto_wrap_policy: Union[str, Callable, None] = None,
        backward_prefetch: Union[str, BackwardPrefetch, None] = None,
        mixed_precision: Union[dict, MixedPrecision, None] = None,
        param_init_fn: Union[str, Callable[[nn.Module], None]] = None,
        use_orig_params: bool = True,
        **kwargs,
    ):
        if isinstance(sharding_strategy, str):
            sharding_strategy = ShardingStrategy[sharding_strategy]
        if not (isinstance(sharding_strategy, ShardingStrategy)
                or sharding_strategy is None):
            raise TypeError(
                'sharding_strategy must be str or enum of `ShardingStrategy` '
                f', but got {sharding_strategy}')

        if isinstance(cpu_offload, bool):
            cpu_offload = CPUOffload(offload_params=cpu_offload)
        if not (isinstance(cpu_offload, CPUOffload) or cpu_offload is None):
            raise TypeError(
                '`cpu_offload` should be `None`, `bool`'
                f'or `CPUOffload`, but has type {type(cpu_offload)}')

        if isinstance(auto_wrap_policy, str):
            auto_wrap_policy = FUNCTIONS.get(  # type: ignore
                auto_wrap_policy)
            if auto_wrap_policy is None:
                raise ValueError('`auto_wrap_policy` is not registered!')

        elif isinstance(auto_wrap_policy, dict):
            policy = auto_wrap_policy.pop('type')
            if isinstance(policy, str):
                
                # NOTE(julieta) special handling for transformer_auto_wrap_policy
                if policy == 'torch.distributed.fsdp.wrap.transformer_auto_wrap_policy':
                    transformer_layer_cls = auto_wrap_policy.pop('transformer_layer_cls')
                    # TODO(julieta) support multiple classes
                    auto_wrap_policy['transformer_layer_cls'] = (FUNCTIONS.get(transformer_layer_cls),)

                policy = FUNCTIONS.get(policy)  # type: ignore

            if policy is None:
                raise ValueError('`auto_wrap_policy` is not registered!')
            auto_wrap_policy = partial(policy, **auto_wrap_policy)

        if not (auto_wrap_policy is None
                or callable(auto_wrap_policy)):  # type: ignore
            raise TypeError('`auto_wrap_policy` should be a str, a '
                            'callable, a dict or None, but has type '
                            f'{type(auto_wrap_policy)}')

        if isinstance(backward_prefetch, str):
            backward_prefetch = BackwardPrefetch[backward_prefetch]
        if not (isinstance(backward_prefetch, BackwardPrefetch)
                or backward_prefetch is None):
            raise TypeError(
                '`backward_prefetch` should be `None`, string of '
                '"BACKWARD_PRE" and "BACKWARD_POST", or '
                f'`BackwardPrefetch`, but has type {type(backward_prefetch)}')

        if isinstance(param_init_fn, str):
            param_init_fn = FUNCTIONS.get(  # type: ignore
                param_init_fn)
            if param_init_fn is None:
                raise ValueError('`param_init_fn` is not registered!')
        elif isinstance(param_init_fn, dict):
            init_fn = param_init_fn.pop('type')
            if isinstance(param_init_fn, str):
                init_fn = FUNCTIONS.get(init_fn)  # type: ignore
            if init_fn is None:
                raise ValueError('`param_init_fn` is not registered!')
            param_init_fn = partial(init_fn, **param_init_fn)

        if not (callable(param_init_fn) or param_init_fn is None):
            raise TypeError('`param_init_fn` should be a str, a '
                            'callable, a dict or None, but has type '
                            f'{type(param_init_fn)}')

        def parse_dtype(dtype):
            if dtype is None:
                return None
            elif isinstance(dtype, str):
                return getattr(torch, dtype)
            elif isinstance(dtype, torch.dtype):
                return dtype
            else:
                raise TypeError(
                    '`dtype` should be `None`, `str` or `torch.dtype`, '
                    f'but has type {type(dtype)}')

        if isinstance(mixed_precision, dict):
            mixed_precision['param_dtype'] = parse_dtype(
                mixed_precision.get('param_dtype', None))
            mixed_precision['reduce_dtype'] = parse_dtype(
                mixed_precision.get('reduce_dtype', None))
            mixed_precision['buffer_dtype'] = parse_dtype(
                mixed_precision.get('buffer_dtype', None))
            mixed_precision = MixedPrecision(**mixed_precision)
        elif isinstance(mixed_precision, MixedPrecision):
            mixed_precision = mixed_precision
        elif mixed_precision is not None:
            raise TypeError(
                '`mixed_precision` should be `None`, `dict` or '
                f'`MixedPrecision`, but has type {type(mixed_precision)}')

        # ignored_parameters and ignored_modules will be deprecated by PyTorch.
        # Therefore we hide them in **kwargs.
        # TODO: Update when PyTorch 2.1.0 released
        if 'ignored_parameters' in kwargs:
            kwargs['ignored_parameters'] = self._get_ignored_params(
                module, kwargs['ignored_parameters'])

        if 'ignored_modules' in kwargs:
            kwargs['ignored_modules'] = self._get_ignored_modules(
                module, kwargs['ignored_modules'])

        super().__init__(
            module=module,
            process_group=process_group,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            use_orig_params=use_orig_params,
            **kwargs)

    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
            call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict): Data sampled by dataloader.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            if isinstance(data, dict):
                losses = self(**data, mode='loss')
            elif isinstance(data, (list, tuple)):
                losses = self(*data, mode='loss')
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                f'list tuple or dict, but got {type(data)}')

        preds = None
        masks = None

        ## for mmpretrain
        if isinstance(losses, tuple) and len(losses) == 3:
            losses, preds, masks = losses

        ## mmpose and mmseg
        elif isinstance(losses, tuple) and len(losses) == 2:
            losses, preds = losses

        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)

        ## mmpretrain
        if preds is not None and masks is not None:
            log_vars['vis_preds'] = preds
            log_vars['vis_masks'] = masks

        ## mmpose and mmseg
        elif preds is not None:
            log_vars['vis_preds'] = preds

        return log_vars

    def val_step(self, data: dict) -> List[BaseDataElement]:
        """Gets the prediction of module during validation process.

        Args:
            data (dict): Data sampled by dataloader.

        Returns:
            List[BaseDataElement] or dict: The predictions of given data.
        """
        data = self.module.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: dict) -> List[BaseDataElement]:
        """Gets the predictions of module during testing process.

        Args:
            data (dict): Data sampled by dataloader.

        Returns:
            List[BaseDataElement]: The predictions of given data.
        """
        data = self.module.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`
        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.
        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def _get_ignored_params(self, module: nn.Module,
                            ignored_parameters: Union[Iterable[str],
                                                      Iterable[nn.Module]]):
        """Get params from string."""
        params_dict = dict(module.named_parameters())
        if is_seq_of(ignored_parameters, str):
            ignored_parameters = [
                params_dict[name] for name in ignored_parameters
            ]
        if not is_seq_of(ignored_parameters,
                         nn.Parameter) and ignored_parameters is not None:
            raise TypeError(
                '`ignored_modules` should be `None`, `Iterable[str]` or '
                '`Iterable[nn.Parameters]`, but has type '
                f'{type(ignored_parameters)}')
        return ignored_parameters

    def _get_ignored_modules(self, module: nn.Module,
                             ignored_modules: Union[Iterable[str],
                                                    Iterable[nn.Module]]):
        """Get modules from string."""
        modules_dict = dict(module.named_modules())
        if is_seq_of(ignored_modules, str):
            ignored_modules = [modules_dict[name] for name in ignored_modules]
        if not is_seq_of(ignored_modules,
                         nn.Module) and ignored_modules is not None:
            raise TypeError(
                '`ignored_modules` should be `None`, `Iterable[str]` or '
                '`Iterable[nn.Module]`, but has type '
                f'{type(ignored_modules)}')
        return ignored_modules

    if digit_version(torch.__version__) < digit_version('2.0.1'):

        @staticmethod
        def optim_state_dict(
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            group: Optional[dist.ProcessGroup] = None,
        ) -> Dict[str, Any]:
            """copied from pytorch 2.0.1 which has fixed some bugs."""
            state_dict_settings = FullyShardedDataParallel.get_state_dict_type(
                model)
            return FullyShardedDataParallel._optim_state_dict_impl(
                model=model,
                optim=optim,
                optim_state_dict=optim.state_dict(),
                optim_input=None,
                rank0_only=getattr(state_dict_settings.optim_state_dict_config,
                                   'rank0_only', False),
                full_state_dict=state_dict_settings.state_dict_type ==
                StateDictType.FULL_STATE_DICT,
                group=group,
            )

        @staticmethod
        def set_state_dict_type(
            module: nn.Module,
            state_dict_type: StateDictType,
            state_dict_config: Optional[StateDictConfig] = None,
            optim_state_dict_config: Optional[OptimStateDictConfig] = None,
        ) -> StateDictSettings:
            """copied from pytorch 2.0.1 which has fixed some bugs."""
            import torch.distributed.fsdp._traversal_utils as traversal_utils
            _state_dict_type_to_config = {
                StateDictType.FULL_STATE_DICT: FullStateDictConfig,
                StateDictType.LOCAL_STATE_DICT: LocalStateDictConfig,
                StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
            }
            _optim_state_dict_type_to_config = {
                StateDictType.FULL_STATE_DICT: FullOptimStateDictConfig,
                StateDictType.LOCAL_STATE_DICT: LocalOptimStateDictConfig,
                StateDictType.SHARDED_STATE_DICT: ShardedOptimStateDictConfig,
            }

            # Use the default config if a state_dict config is not set.
            state_dict_config_type = _state_dict_type_to_config[
                state_dict_type]
            optim_state_dict_config_type = _optim_state_dict_type_to_config[
                state_dict_type]
            if state_dict_config is None:
                state_dict_config = state_dict_config_type()
            if optim_state_dict_config is None:
                optim_state_dict_config = optim_state_dict_config_type()
            if state_dict_config_type != type(state_dict_config):
                raise RuntimeError('Expected state_dict_config of type '
                                   f'{state_dict_config_type} '
                                   f'but got {type(state_dict_config)}')
            if optim_state_dict_config_type != type(optim_state_dict_config):
                raise RuntimeError('Expected optim_state_dict_config of type '
                                   f'{optim_state_dict_config_type} '
                                   f'but got {type(optim_state_dict_config)}')

            # Set the state_dict type and configurations.
            prev_state_dict_type = None
            prev_state_dict_config = None
            prev_optim_state_dict_config = None
            for submodule in traversal_utils._get_fsdp_states(module):
                if prev_state_dict_type is None:
                    prev_state_dict_type = submodule._state_dict_type
                else:
                    assert (
                        prev_state_dict_type == submodule._state_dict_type
                    ), 'All FSDP modules should have the same state_dict_type.'
                if prev_state_dict_config is None:
                    prev_state_dict_config = submodule._state_dict_config
                else:
                    assert isinstance(
                        submodule._state_dict_config,
                        type(prev_state_dict_config)), (
                            'All FSDP modules must have the same type of '
                            'state_dict_config.')
                if prev_optim_state_dict_config is None:
                    prev_optim_state_dict_config = \
                        submodule._optim_state_dict_config
                else:
                    assert isinstance(
                        submodule._optim_state_dict_config,
                        type(prev_optim_state_dict_config),
                    ), ('All FSDP modules must have the same type of '
                        'optim_state_dict_config.')

                submodule._state_dict_type = state_dict_type
                submodule._state_dict_config = state_dict_config
                submodule._optim_state_dict_config = optim_state_dict_config

            return StateDictSettings(prev_state_dict_type,
                                     prev_state_dict_config,
                                     prev_optim_state_dict_config)
