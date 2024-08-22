# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement


class BaseClassifier(BaseModel, metaclass=ABCMeta):
    """Base class for classifiers.

    Args:
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None, it will use "BaseDataPreprocessor" as type, see
            :class:`mmengine.model.BaseDataPreprocessor` for more details.
            Defaults to None.

    Attributes:
        init_cfg (dict): Initialization config dict.
        data_preprocessor (:obj:`mmengine.model.BaseDataPreprocessor`): An
            extra data pre-processing module, which processes data from
            dataloader to the format accepted by :meth:`forward`.
    """

    def __init__(self,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(BaseClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

    @property
    def with_neck(self) -> bool:
        """Whether the classifier has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Whether the classifier has a head."""
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`BaseDataElement`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...)
                in general.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmengine.BaseDataElement`.
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

    def extract_feats(self, multi_inputs: Sequence[torch.Tensor],
                      **kwargs) -> list:
        """Extract features from a sequence of input tensor.

        Args:
            multi_inputs (Sequence[torch.Tensor]): A sequence of input
                tensor. It can be used in augmented inference.
            **kwargs: Other keyword arguments accepted by :meth:`extract_feat`.

        Returns:
            list: Features of every input tensor.
        """
        assert isinstance(multi_inputs, Sequence), \
            '`extract_feats` is used for a sequence of inputs tensor. If you '\
            'want to extract on single inputs tensor, use `extract_feat`.'
        return [self.extract_feat(inputs, **kwargs) for inputs in multi_inputs]
