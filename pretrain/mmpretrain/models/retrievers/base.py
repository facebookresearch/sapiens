# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch.utils.data import DataLoader


class BaseRetriever(BaseModel, metaclass=ABCMeta):
    """Base class for retriever.

    Args:
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None, it will use "BaseDataPreprocessor" as type, see
            :class:`mmengine.model.BaseDataPreprocessor` for more details.
            Defaults to None.
        prototype (Union[DataLoader, dict, str, torch.Tensor]): Database to be
            retrieved. The following four types are supported.

            - DataLoader: The original dataloader serves as the prototype.
            - dict: The configuration to construct Dataloader.
            - str: The path of the saved vector.
            - torch.Tensor: The saved tensor whose dimension should be dim.

    Attributes:
        prototype (Union[DataLoader, dict, str, torch.Tensor]): Database to be
            retrieved. The following four types are supported.

            - DataLoader: The original dataloader serves as the prototype.
            - dict: The configuration to construct Dataloader.
            - str: The path of the saved vector.
            - torch.Tensor: The saved tensor whose dimension should be dim.

        data_preprocessor (:obj:`mmengine.model.BaseDataPreprocessor`): An
            extra data pre-processing module, which processes data from
            dataloader to the format accepted by :meth:`forward`.
    """

    def __init__(
        self,
        prototype: Union[DataLoader, dict, str, torch.Tensor] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        super(BaseRetriever, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.prototype = prototype
        self.prototype_inited = False

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'loss'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor without any
          post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor, tuple): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        pass

    def extract_feat(self, inputs: torch.Tensor):
        """Extract features from the input tensor with shape (N, C, ...).

        The sub-classes are recommended to implement this method to extract
        features from backbone and neck.

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
        """
        raise NotImplementedError

    def loss(self, inputs: torch.Tensor,
             data_samples: List[BaseDataElement]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        raise NotImplementedError

    def predict(self,
                inputs: tuple,
                data_samples: Optional[List[BaseDataElement]] = None,
                **kwargs) -> List[BaseDataElement]:
        """Predict results from the extracted features.

        Args:
            inputs (tuple): The features extracted from the backbone.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        raise NotImplementedError

    def matching(self, inputs: torch.Tensor):
        """Compare the prototype and calculate the similarity.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C).
        """
        raise NotImplementedError

    def prepare_prototype(self):
        """Preprocessing the prototype before predict."""
        raise NotImplementedError

    def dump_prototype(self, path):
        """Save the features extracted from the prototype to the specific path.

        Args:
            path (str): Path to save feature.
        """
        raise NotImplementedError
