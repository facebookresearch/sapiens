# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, contextmanager
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.device import get_device
from mmengine.optim import OptimWrapperDict
from mmengine.registry import MODEL_WRAPPERS
from .distributed import MMDistributedDataParallel


@MODEL_WRAPPERS.register_module()
class MMSeparateDistributedDataParallel(DistributedDataParallel):
    """A DistributedDataParallel wrapper for models in MMGeneration.

    In MMedting and MMGeneration there is a need to wrap different modules in
    the models with separate DistributedDataParallel. Otherwise, it will cause
    errors for GAN training. For example, the GAN model, usually has two
    submodules: generator and discriminator. If we wrap both of them in one
    standard DistributedDataParallel, it will cause errors during training,
    because when we update the parameters of the generator (or discriminator),
    the parameters of the discriminator (or generator) is not updated, which is
    not allowed for DistributedDataParallel. So we design this wrapper to
    separately wrap DistributedDataParallel for generator and discriminator.
    In this wrapper, we perform two operations:

    1. Wraps each module in the models with separate MMDistributedDataParallel.
       Note that only modules with parameters will be wrapped.
    2. Calls ``train_step``, ``val_step`` and ``test_step`` of submodules to
       get losses and predictions.

    Args:
        module (nn.Module): model contain multiple submodules which have
            separately updating strategy.
        broadcast_buffers (bool): Same as that in
            ``torch.nn.parallel.distributed.DistributedDataParallel``.
            Defaults to False.
        find_unused_parameters (bool): Same as that in
            ``torch.nn.parallel.distributed.DistributedDataParallel``.
            Traverse the autograd graph of all tensors contained in returned
            value of the wrapped module's forward function. Defaults to False.
        **kwargs: Keyword arguments passed to ``MMDistributedDataParallel``.

            - device_ids (List[int] or torch.device, optional): CUDA devices
              for module.
            - output_device (int or torch.device, optional): Device location of
              output for single-device CUDA modules.
            - dim (int): Defaults to 0.
            - process_group (ProcessGroup, optional): The process group to be
              used for distributed data all-reduction.
            - bucket_cap_mb (int): bucket size in MegaBytes (MB). Defaults
              to 25.
            - check_reduction (bool): This argument is deprecated. Defaults
              to False.
            - gradient_as_bucket_view (bool): Defaults to False.
            - static_graph (bool): Defaults to False.

    See more information about arguments in
    :class:`torch.nn.parallel.DistributedDataParallel`.
    """

    def __init__(self,
                 module: nn.Module,
                 broadcast_buffers: bool = False,
                 find_unused_parameters: bool = False,
                 **kwargs):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        device = get_device()
        # Wrap the submodule with parameters of `self.module` to
        # `MMDistributedDataParallel`
        for name, sub_module in module._modules.items():
            # module without parameters.
            if next(sub_module.parameters(), None) is None:
                sub_module = sub_module.to(device)
            elif all(not p.requires_grad for p in sub_module.parameters()):
                sub_module = sub_module.to(device)
            else:
                sub_module = MMDistributedDataParallel(
                    module=sub_module.to(device),
                    broadcast_buffers=broadcast_buffers,
                    find_unused_parameters=find_unused_parameters,
                    **kwargs)
            module._modules[name] = sub_module

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapperDict): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A dict of tensor for logging.
        """
        return self.module.train_step(data, optim_wrapper)

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.val_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.test_step(data)

    @contextmanager
    def no_sync(self):
        """Enables ``no_sync`` context of all sub ``MMDistributedDataParallel``
        modules."""
        with ExitStack() as stack:
            for sub_ddp_model in self.module._modules.values():
                stack.enter_context(sub_ddp_model.no_sync())
                yield

    def train(self, mode: bool = True) -> 'MMSeparateDistributedDataParallel':
        """Sets the module in training mode.

        In order to make the ddp wrapper inheritance hierarchy more uniform,
        ``MMSeparateDistributedDataParallel`` inherits from
        ``DistributedDataParallel``, but will not call its constructor.
        Since the attributes of ``DistributedDataParallel`` have not been
        initialized, call the ``train`` method of ``DistributedDataParallel``
        will raise an error if pytorch version <= 1.9. Therefore, override
        this method to call the ``train`` method of submodules.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                mode (``False``). Defaults to ``True``.

        Returns:
            Module: self.
        """
        self.training = mode
        self.module.train(mode)
        return self
