# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmengine.utils import is_list_of
from ..base_module import BaseModule
from .data_preprocessor import BaseDataPreprocessor


class BaseModel(BaseModule):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.

    Subclasses inherit from BaseModel only need to implement the forward
    method, which implements the logic to calculate loss and predictions,
    then can be trained in the runner.

    Examples:
        >>> @MODELS.register_module()
        >>> class ToyModel(BaseModel):
        >>>
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.backbone = nn.Sequential()
        >>>         self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
        >>>         self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
        >>>         self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
        >>>         self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        >>>         self.backbone.add_module('fc2', nn.Linear(120, 84))
        >>>         self.backbone.add_module('fc3', nn.Linear(84, 10))
        >>>
        >>>         self.criterion = nn.CrossEntropyLoss()
        >>>
        >>>     def forward(self, batch_inputs, data_samples, mode='tensor'):
        >>>         data_samples = torch.stack(data_samples)
        >>>         if mode == 'tensor':
        >>>             return self.backbone(batch_inputs)
        >>>         elif mode == 'predict':
        >>>             feats = self.backbone(batch_inputs)
        >>>             predictions = torch.argmax(feats, 1)
        >>>             return predictions
        >>>         elif mode == 'loss':
        >>>             feats = self.backbone(batch_inputs)
        >>>             loss = self.criterion(feats, data_samples)
        >>>             return dict(loss=loss)

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if data_preprocessor is None:
            data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        return loss, log_vars  # type: ignore

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.
        if args and isinstance(args[0], str) and 'npu' in args[0]:
            import torch_npu
            args = tuple([
                list(args)[0].replace(
                    'npu', torch_npu.npu.native_device if hasattr(
                        torch_npu.npu, 'native_device') else 'privateuseone')
            ])
        if kwargs and 'npu' in str(kwargs.get('device', '')):
            import torch_npu
            kwargs['device'] = kwargs['device'].replace(
                'npu', torch_npu.npu.native_device if hasattr(
                    torch_npu.npu, 'native_device') else 'privateuseone')

        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._set_device(torch.device(device))
        return super().to(*args, **kwargs)

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device('cuda', index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def mlu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.mlu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        device = torch.device('mlu', torch.mlu.current_device())
        self._set_device(device)
        return super().mlu()

    def npu(
        self,
        device: Union[int, str, torch.device, None] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.npu`
        additionally.

        Returns:
            nn.Module: The model itself.

        Note:
            This generation of NPU(Ascend910) does not support
            the use of multiple cards in a single process,
            so the index here needs to be consistent with the default device
        """
        device = torch.npu.current_device()
        self._set_device(device)
        return super().npu()

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device('cpu'))
        return super().cpu()

    def _set_device(self, device: torch.device) -> None:
        """Recursively set device for `BaseDataPreprocessor` instance.

        Args:
            device (torch.device): the desired device of the parameters and
                buffers in this module.
        """

        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor):
                return
            if device is not None:
                module._device = device

        self.apply(apply_fn)

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``batch_inputs`` and ``data_sample`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.test_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (list, optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of results used for computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            dict or list:
                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of inference
                  results.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` of tensor for custom use.
        """

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
