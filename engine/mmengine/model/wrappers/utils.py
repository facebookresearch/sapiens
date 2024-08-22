# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from mmengine.registry import MODEL_WRAPPERS, Registry


def is_model_wrapper(model: nn.Module, registry: Registry = MODEL_WRAPPERS):
    """Check if a module is a model wrapper.

    The following 4 model in MMEngine (and their subclasses) are regarded as
    model wrappers: DataParallel, DistributedDataParallel,
    MMDataParallel, MMDistributedDataParallel. You may add you own
    model wrapper by registering it to ``mmengine.registry.MODEL_WRAPPERS``.

    Args:
        model (nn.Module): The model to be checked.
        registry (Registry): The parent registry to search for model wrappers.

    Returns:
        bool: True if the input model is a model wrapper.
    """
    module_wrappers = tuple(registry.module_dict.values())
    if isinstance(model, module_wrappers):
        return True

    if not registry.children:
        return False

    return any(
        is_model_wrapper(model, child) for child in registry.children.values())
