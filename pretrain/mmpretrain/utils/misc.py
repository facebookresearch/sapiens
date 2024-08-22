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
