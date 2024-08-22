# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Optional

import transformers
from mmengine.registry import Registry
from transformers import AutoConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from mmpretrain.registry import MODELS, TOKENIZER


def register_hf_tokenizer(
    cls: Optional[type] = None,
    registry: Registry = TOKENIZER,
):
    """Register HuggingFace-style PreTrainedTokenizerBase class."""
    if cls is None:

        # use it as a decorator: @register_hf_tokenizer()
        def _register(cls):
            register_hf_tokenizer(cls=cls)
            return cls

        return _register

    def from_pretrained(**kwargs):
        if ('pretrained_model_name_or_path' not in kwargs
                and 'name_or_path' not in kwargs):
            raise TypeError(
                f'{cls.__name__}.from_pretrained() missing required '
                "argument 'pretrained_model_name_or_path' or 'name_or_path'.")
        # `pretrained_model_name_or_path` is too long for config,
        # add an alias name `name_or_path` here.
        name_or_path = kwargs.pop('pretrained_model_name_or_path',
                                  kwargs.pop('name_or_path'))
        return cls.from_pretrained(name_or_path, **kwargs)

    registry._register_module(module=from_pretrained, module_name=cls.__name__)
    return cls


_load_hf_pretrained_model = True


@contextlib.contextmanager
def no_load_hf_pretrained_model():
    global _load_hf_pretrained_model
    _load_hf_pretrained_model = False
    yield
    _load_hf_pretrained_model = True


def register_hf_model(
    cls: Optional[type] = None,
    registry: Registry = MODELS,
):
    """Register HuggingFace-style PreTrainedModel class."""
    if cls is None:

        # use it as a decorator: @register_hf_tokenizer()
        def _register(cls):
            register_hf_model(cls=cls)
            return cls

        return _register

    if issubclass(cls, _BaseAutoModelClass):
        get_config = AutoConfig.from_pretrained
        from_config = cls.from_config
    elif issubclass(cls, PreTrainedModel):
        get_config = cls.config_class.from_pretrained
        from_config = cls
    else:
        raise TypeError('Not auto model nor pretrained model of huggingface.')

    def build(**kwargs):
        if ('pretrained_model_name_or_path' not in kwargs
                and 'name_or_path' not in kwargs):
            raise TypeError(
                f'{cls.__name__} missing required argument '
                '`pretrained_model_name_or_path` or `name_or_path`.')
        # `pretrained_model_name_or_path` is too long for config,
        # add an alias name `name_or_path` here.
        name_or_path = kwargs.pop('pretrained_model_name_or_path',
                                  kwargs.pop('name_or_path'))

        if kwargs.pop('load_pretrained', True) and _load_hf_pretrained_model:
            model = cls.from_pretrained(name_or_path, **kwargs)
            setattr(model, 'is_init', True)
            return model
        else:
            cfg = get_config(name_or_path, **kwargs)
            return from_config(cfg)

    registry._register_module(module=build, module_name=cls.__name__)
    return cls


register_hf_model(transformers.AutoModelForCausalLM)
