# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import torch
from mmengine.model import BaseModel
from torch import nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


class BaseSelfSupervisor(BaseModel, metaclass=ABCMeta):
    """BaseModel for Self-Supervised Learning.

    All self-supervised algorithms should inherit this module.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        target_generator: (dict, optional): The target_generator module to
            generate targets for self-supervised learning optimization, such as
            HOG, extracted features from other modules(DALL-E, CLIP), etc.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (Union[dict, nn.Module], optional): The config for
            preprocessing input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 target_generator: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'SelfSupDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)
        if target_generator is not None and not isinstance(
                target_generator, nn.Module):
            target_generator = MODELS.build(target_generator)

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.target_generator = target_generator

        return

    @property
    def with_neck(self) -> bool:
        """Check if the model has a neck module."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Check if the model has a head module."""
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_target_generator(self) -> bool:
        """Check if the model has a target_generator module."""
        return hasattr(
            self, 'target_generator') and self.target_generator is not None

    def forward(self,
                inputs: Union[torch.Tensor, List[torch.Tensor]],
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method currently accepts two modes: "tensor" and "loss":

        - "tensor": Forward the backbone network and return the feature
          tensor(s) tensor without any post-processing, same as a common
          PyTorch Module.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor or List[torch.Tensor]): The input tensor with
                shape (N, C, ...) in general.
            data_samples (List[DataSample], optional): The other data of
                every samples. It's required for some algorithms
                if ``mode="loss"``. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        """Extract features from the input tensor with shape (N, C, ...).

        The default behavior is extracting features from backbone.

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.

        Returns:
            tuple | Tensor: The output feature tensor(s).
        """
        x = self.backbone(inputs)
        return x

    @abstractmethod
    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        raise NotImplementedError

    def get_layer_depth(self, param_name: str):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if hasattr(self.backbone, 'get_layer_depth'):
            return self.backbone.get_layer_depth(param_name, 'backbone.')
        else:
            raise NotImplementedError(
                f"The backbone {type(self.backbone)} doesn't "
                'support `get_layer_depth` by now.')
