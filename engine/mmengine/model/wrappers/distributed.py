# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from ..utils import detect_anomalous_params

MODEL_WRAPPERS.register_module(module=DistributedDataParallel)
MODEL_WRAPPERS.register_module(module=DataParallel)


@MODEL_WRAPPERS.register_module()
class MMDistributedDataParallel(DistributedDataParallel):
    """A distributed model wrapper used for training,testing and validation in
    loop.

    Different from DistributedDataParallel, MMDistributedDataParallel
    implements three methods :meth:`train_step`, :meth:`val_step` and
    :meth:`test_step`, which will be called by ``train_loop``, ``val_loop``
    and ``test_loop``.

    - ``train_step``: Called by ``runner.train_loop``, and implement
      default model forward, gradient back propagation, parameter updating
      logic. To take advantage of DistributedDataParallel's automatic gradient
      synchronization, ``train_step`` calls ``DistributedDataParallel.forward``
      to calculate the losses, and call other methods of :class:`BaseModel` to
      pre-process data and parse losses. Finally, update model parameters by
      :class:`OptimWrapper` and return the loss dictionary used
      for logging.

    - ``val_step``: Called by ``runner.val_loop`` and get the inference
      results. Since there is no gradient synchronization requirement,
      this procedure is equivalent to ``BaseModel.val_step``

    - ``test_step``: Called by ``runner.test_loop``, equivalent ``val_step``.

    Args:
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

            - Parameters were not used during forward pass.
            - Parameters were not used to produce loss.

            Defaults to False.

        **kwargs: keyword arguments passed to ``DistributedDataParallel``.

            - device_ids (List[int] or torch.device, optional): CUDA devices
              for module.
            - output_device (int or torch.device, optional): Device location of
              output for single-device CUDA modules.
            - dim (int): Defaults to 0.
            - broadcast_buffers (bool): Flag that enables syncing (
              broadcasting) buffers of the module at beginning of the
              ``forward`` function. Defaults to True
            - find_unused_parameters (bool): Whether to find parameters of
              module, which are not in the forward graph. Defaults to False.
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

    Note:
        If model has multiple submodules and each module has
        separate optimization strategies,
        :class:`MMSeparateDistributedDataParallel` should be used to wrap
        the model.

    Note:
        If model itself has custom optimization strategy, rather than
        simply forward model and update model. A custom model wrapper
        inherit from ``MMDistributedDataParallel`` should be defined and
        override the ``train_step`` method.
    """

    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(self, data: Union[dict, tuple, list],
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
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            losses = self._run_forward(data, mode='loss')
            preds = None
            masks = None

            ## for mmpretrain
            if isinstance(losses, tuple) and len(losses) == 3:
                losses, preds, masks = losses

            ## for mmpose and mmseg
            elif isinstance(losses, tuple) and len(losses) == 2:
                losses, preds = losses

        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self)

        ## mmpretrain
        if preds is not None and masks is not None:
            log_vars['vis_preds'] = preds
            log_vars['vis_masks'] = masks

        ## mmpose and mmseg
        elif preds is not None:
            log_vars['vis_preds'] = preds

        return log_vars

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

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
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
