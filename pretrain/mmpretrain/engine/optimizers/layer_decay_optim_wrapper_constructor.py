# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Callable, List, Optional

from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm
from torch import nn
from torch.nn import GroupNorm, LayerNorm

from mmpretrain.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone.

    By default, each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``layer_decay_rate`` (float): The learning rate of a parameter will
      multiply it by multiple times according to the layer depth of the
      parameter. Usually, it's less than 1, so that the earlier layers will
      have a lower learning rate. Defaults to 1.
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in normalization layers).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization layers.
    - ``flat_decay_mult`` (float): It will be multiplied to the weight
      decay for all one-dimensional parameters
    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_decay_mult`` will be
      ignored. It should be a dict and may contain fields ``decay_mult``.
      (The ``lr_mult`` is disabled in this constructor).

    Example:

    In the config file, you can use this constructor as below:

    .. code:: python

        optim_wrapper = dict(
            optimizer=dict(
                type='AdamW',
                lr=4e-3,
                weight_decay=0.05,
                eps=1e-8,
                betas=(0.9, 0.999)),
            constructor='LearningRateDecayOptimWrapperConstructor',
            paramwise_cfg=dict(
                layer_decay_rate=0.75,  # layer-wise lr decay factor
                norm_decay_mult=0.,
                flat_decay_mult=0.,
                custom_keys={
                    '.cls_token': dict(decay_mult=0.0),
                    '.pos_embed': dict(decay_mult=0.0)
                }))
    """

    def add_params(self,
                   params: List[dict],
                   module: nn.Module,
                   prefix: str = '',
                   get_layer_depth: Optional[Callable] = None,
                   **kwargs) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (List[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            optimizer_cfg (dict): The configuration of optimizer.
            prefix (str): The prefix of the module.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)
        logger = MMLogger.get_current_instance()

        # The model should have `get_layer_depth` method
        if get_layer_depth is None and not hasattr(module, 'get_layer_depth'):
            raise NotImplementedError('The layer-wise learning rate decay need'
                                      f' the model {type(module)} has'
                                      ' `get_layer_depth` method.')
        else:
            get_layer_depth = get_layer_depth or module.get_layer_depth

        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', None)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', None)
        flat_decay_mult = self.paramwise_cfg.get('flat_decay_mult', None)
        decay_rate = self.paramwise_cfg.get('layer_decay_rate', 1.0)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            param_name = prefix + name
            if not param.requires_grad:
                continue

            if self.base_wd is not None:
                base_wd = self.base_wd
                custom_key = next(
                    filter(lambda k: k in param_name, sorted_keys), None)
                # custom parameters decay
                if custom_key is not None:
                    custom_cfg = custom_keys[custom_key].copy()
                    decay_mult = custom_cfg.pop('decay_mult', 1.)

                    param_group['weight_decay'] = base_wd * decay_mult
                    # add custom settings to param_group
                    param_group.update(custom_cfg)
                # norm decay
                elif is_norm and norm_decay_mult is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
                # bias decay
                elif name == 'bias' and bias_decay_mult is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
                # flatten parameters decay
                elif param.ndim == 1 and flat_decay_mult is not None:
                    param_group['weight_decay'] = base_wd * flat_decay_mult
                else:
                    param_group['weight_decay'] = base_wd

            layer_id, max_id = get_layer_depth(param_name)

            scale = decay_rate**(max_id - layer_id - 1)
            param_group['lr'] = self.base_lr * scale
            param_group['lr_scale'] = scale
            param_group['layer_id'] = layer_id
            param_group['param_name'] = param_name

            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}{child_name}.'
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                get_layer_depth=get_layer_depth,
            )

        if prefix == '':
            layer_params = defaultdict(list)
            for param in params:
                layer_params[param['layer_id']].append(param)
            for layer_id, layer_params in layer_params.items():
                lr_scale = layer_params[0]['lr_scale']
                lr = layer_params[0]['lr']
                msg = [
                    f'layer {layer_id} params '
                    f'(lr={lr:.3g}, lr_scale={lr_scale:.3g}):'
                ]
                for param in layer_params:
                    msg.append(f'\t{param["param_name"]}: '
                               f'weight_decay={param["weight_decay"]:.3g}')
                logger.debug('\n'.join(msg))
