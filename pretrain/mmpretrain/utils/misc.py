# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from mmengine.model import is_model_wrapper


def get_ori_model(model: nn.Module) -> nn.Module:
    """Get original model if the input model is a model wrapper.

    Args:
        model (nn.Module): A model may be a model wrapper.

    Returns:
        nn.Module: The model without model wrapper.
    """
    if is_model_wrapper(model):
        return model.module
    else:
        return model
