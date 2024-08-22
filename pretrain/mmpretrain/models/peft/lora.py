# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from typing import Any, List

import torch
from mmengine.logging import print_log
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS


class LoRALinear(nn.Module):
    r"""Implements LoRA in a linear layer.

    Args:
        original_layer (nn.Linear): The linear layer to be finetuned.
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.

    Note:
        The forward process of LoRA linear layer is:

        .. math::
            `y = W_0 x + BAx * (\alpha / r)`

        Where :math:`x` is the input, :math:`y` is the output,
        :math:`W_0` is the parameter of the original layer,
        :math:`A` and :math:`B` are the low-rank decomposition matrixs,
        :math: `\alpha` is the scale factor and :math: `r` is the rank.
    """

    def __init__(self,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super(LoRALinear, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling

        return out + lora_out


@MODELS.register_module()
class LoRAModel(BaseModule):
    """Implements LoRA in a module.

    An PyTorch implement of : `LoRA: Low-Rank Adaptation
    of Large Language Models <https://arxiv.org/abs/2106.09685>`_

    Args:
        module (dict): The config of the module to be finetuned. See
            :mod:`mmpretrain.models`
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.
        targets (List[dict]): The target layers to be applied with the LoRA.
            Defaults to a empty list. Specify by regular expression or suffix.

    Examples:
        >>> model = LoRAModel(
        ...     module=dict(type='VisionTransformer', arch='b'),
        ...     alpha=4,
        ...     rank=4,
        ...     drop_rate=0.1,
        ...     targets=[
        ...         dict(type='.*qkv'), # regular expression
        ...         dict(type='proj', alpha=8, rank=8), # suffix
        ...     ])
    """

    def __init__(self,
                 module: dict,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.,
                 targets: List[dict] = list()):

        super().__init__()

        module = MODELS.build(module)
        module.init_weights()

        self.module = module
        self.alpha = alpha
        self.rank = rank
        self.drop_rate = drop_rate

        assert len(targets) != 0, \
            'The length of target layers should not be 0.'

        self.targets = targets

        self.applied = False
        self.apply_lora()

        if not self.applied:
            raise ValueError(
                'No lora layer is replaced. Please check targets.')

        self._set_lora_trainable()
        self._register_state_dict_hooks()

    def apply_lora(self):
        """Apply LoRA to target layers."""
        module_names = [k for k, _ in self.module.named_modules()]
        for module_name in module_names:
            for target in self.targets:
                target_name = target['type']
                target_alpha = target.get('alpha', self.alpha)
                target_rank = target.get('rank', self.rank)
                target_drop_rate = target.get('drop_rate', self.drop_rate)

                if re.fullmatch(target_name, module_name) or \
                        module_name.endswith(target_name):
                    current_module = self.module.get_submodule(module_name)
                    if isinstance(current_module, nn.Linear):
                        print_log(
                            f'Set LoRA for {module_name} '
                            f'with alpha: {target_alpha}, '
                            f'rank: {target_rank}, '
                            f'drop rate: {target_drop_rate}',
                            logger='current')

                        self._replace_module(module_name, current_module,
                                             target_alpha, target_rank,
                                             target_drop_rate)
                        self.applied = True

    def _replace_module(self, module_name: str, current_module: nn.Module,
                        alpha: int, rank: int, drop_rate: float):
        """Replace target layer with LoRA linear layer in place."""
        parent_module_name = '.'.join(module_name.split('.')[:-1])
        parent_module = self.module.get_submodule(parent_module_name)

        target_name = module_name.split('.')[-1]
        target_module = LoRALinear(current_module, alpha, rank, drop_rate)
        setattr(parent_module, target_name, target_module)

    def _set_lora_trainable(self):
        """Set only the lora parameters trainable."""
        for name, param in self.named_parameters():
            if '.lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _register_state_dict_hooks(self):
        """Register state dict hooks.

        Register state dict saving hooks to save only the lora parameters to
        the state dict. And register state dict loading hooks to handle the
        incompatible keys while loading the state dict.
        """

        def _state_dict_hook(module, state_dict, prefix, local_metadata):
            """Save only the lora parameters to the state dict."""
            keys = [k for k, _ in state_dict.items()]
            for key in keys:
                if '.lora_' not in key:
                    state_dict.pop(key)

        self._register_state_dict_hook(_state_dict_hook)

        def _load_state_dict_post_hook(module, incompatible_keys):
            """Handle the incompatible keys while loading the state dict."""
            missing_keys = incompatible_keys.missing_keys.copy()
            for key in missing_keys:
                if '.lora_' not in key:
                    incompatible_keys.missing_keys.remove(key)

            unexpected_keys = incompatible_keys.unexpected_keys.copy()
            for key in unexpected_keys:
                if '.lora_' not in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super(LoRAModel, self).__getattr__(name)
        except AttributeError:
            return self.module.__getattribute__(name)
