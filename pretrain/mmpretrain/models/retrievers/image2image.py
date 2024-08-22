# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union

import mmengine.dist as dist
import torch
import torch.nn as nn
from mmengine.runner import Runner
from torch.utils.data import DataLoader

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.utils import track_on_main_process
from .base import BaseRetriever


@MODELS.register_module()
class ImageToImageRetriever(BaseRetriever):
    """Image To Image Retriever for supervised retrieval task.

    Args:
        image_encoder (Union[dict, List[dict]]): Encoder for extracting
            features.
        prototype (Union[DataLoader, dict, str, torch.Tensor]): Database to be
            retrieved. The following four types are supported.

            - DataLoader: The original dataloader serves as the prototype.
            - dict: The configuration to construct Dataloader.
            - str: The path of the saved vector.
            - torch.Tensor: The saved tensor whose dimension should be dim.

        head (dict, optional): The head module to calculate loss from
            processed features. See :mod:`mmpretrain.models.heads`. Notice
            that if the head is not set, `loss` method cannot be used.
            Defaults to None.
        similarity_fn (Union[str, Callable]): The way that the similarity
            is calculated. If `similarity` is callable, it is used directly
            as the measure function. If it is a string, the appropriate
            method will be used.  The larger the calculated value, the
            greater the similarity. Defaults to "cosine_similarity".
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        topk (int): Return the topk of the retrieval result. `-1` means
            return all. Defaults to -1.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 image_encoder: Union[dict, List[dict]],
                 prototype: Union[DataLoader, dict, str, torch.Tensor],
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 similarity_fn: Union[str, Callable] = 'cosine_similarity',
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 topk: int = -1,
                 init_cfg: Optional[dict] = None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmpretrain.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(ImageToImageRetriever, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(image_encoder, nn.Module):
            image_encoder = MODELS.build(image_encoder)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.image_encoder = image_encoder
        self.head = head

        self.similarity = similarity_fn

        assert isinstance(prototype, (str, torch.Tensor, dict, DataLoader)), (
            'The `prototype` in  `ImageToImageRetriever` must be a path, '
            'a torch.Tensor, a dataloader or a dataloader dict format config.')
        self.prototype = prototype
        self.prototype_inited = False
        self.topk = topk

    @property
    def similarity_fn(self):
        """Returns a function that calculates the similarity."""
        # If self.similarity_way is callable, return it directly
        if isinstance(self.similarity, Callable):
            return self.similarity

        if self.similarity == 'cosine_similarity':
            # a is a tensor with shape (N, C)
            # b is a tensor with shape (M, C)
            # "cosine_similarity" will get the matrix of similarity
            # with shape (N, M).
            # The higher the score is, the more similar is
            return lambda a, b: torch.cosine_similarity(
                a.unsqueeze(1), b.unsqueeze(0), dim=-1)
        else:
            raise RuntimeError(f'Invalid function "{self.similarity_fn}".')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
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
        if mode == 'tensor':
            return self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs):
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
        Returns:
            Tensor: The output of encoder.
        """

        feat = self.image_encoder(inputs)
        return feat

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def matching(self, inputs: torch.Tensor):
        """Compare the prototype and calculate the similarity.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C).
        Returns:
            dict: a dictionary of score and prediction label based on fn.
        """
        sim = self.similarity_fn(inputs, self.prototype_vecs)
        sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
        predictions = dict(
            score=sim, pred_label=indices, pred_score=sorted_sim)
        return predictions

    def predict(self,
                inputs: tuple,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from the extracted features.

        Args:
            inputs (tuple): The features extracted from the backbone.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        Returns:
            List[DataSample]: the raw data_samples with
                the predicted results
        """
        if not self.prototype_inited:
            self.prepare_prototype()

        feats = self.extract_feat(inputs)
        if isinstance(feats, tuple):
            feats = feats[-1]

        # Matching of similarity
        result = self.matching(feats)
        return self._get_predictions(result, data_samples)

    def _get_predictions(self, result, data_samples):
        """Post-process the output of retriever."""
        pred_scores = result['score']
        pred_labels = result['pred_label']
        if self.topk != -1:
            topk = min(self.topk, pred_scores.size()[-1])
            pred_labels = pred_labels[:, :topk]

        if data_samples is not None:
            for data_sample, score, label in zip(data_samples, pred_scores,
                                                 pred_labels):
                data_sample.set_pred_score(score).set_pred_label(label)
        else:
            data_samples = []
            for score, label in zip(pred_scores, pred_labels):
                data_samples.append(
                    DataSample().set_pred_score(score).set_pred_label(label))
        return data_samples

    def _get_prototype_vecs_from_dataloader(self, data_loader):
        """get prototype_vecs from dataloader."""
        self.eval()
        num = len(data_loader.dataset)

        prototype_vecs = None
        for data_batch in track_on_main_process(data_loader,
                                                'Prepare prototype'):
            data = self.data_preprocessor(data_batch, False)
            feat = self(**data)
            if isinstance(feat, tuple):
                feat = feat[-1]

            if prototype_vecs is None:
                dim = feat.shape[-1]
                prototype_vecs = torch.zeros(num, dim)
            for i, data_sample in enumerate(data_batch['data_samples']):
                sample_idx = data_sample.get('sample_idx')
                prototype_vecs[sample_idx] = feat[i]

        assert prototype_vecs is not None
        dist.all_reduce(prototype_vecs)
        return prototype_vecs

    def _get_prototype_vecs_from_path(self, proto_path):
        """get prototype_vecs from prototype path."""
        data = [None]
        if dist.is_main_process():
            data[0] = torch.load(proto_path)
        dist.broadcast_object_list(data, src=0)
        prototype_vecs = data[0]
        assert prototype_vecs is not None
        return prototype_vecs

    @torch.no_grad()
    def prepare_prototype(self):
        """Used in meta testing. This function will be called before the meta
        testing. Obtain the vector based on the prototype.

        - torch.Tensor: The prototype vector is the prototype
        - str: The path of the extracted feature path, parse data structure,
            and generate the prototype feature vector set
        - Dataloader or config: Extract and save the feature vectors according
            to the dataloader
        """
        device = next(self.image_encoder.parameters()).device
        if isinstance(self.prototype, torch.Tensor):
            prototype_vecs = self.prototype
        elif isinstance(self.prototype, str):
            prototype_vecs = self._get_prototype_vecs_from_path(self.prototype)
        elif isinstance(self.prototype, (dict, DataLoader)):
            loader = Runner.build_dataloader(self.prototype)
            prototype_vecs = self._get_prototype_vecs_from_dataloader(loader)

        self.register_buffer(
            'prototype_vecs', prototype_vecs.to(device), persistent=False)
        self.prototype_inited = True

    def dump_prototype(self, path):
        """Save the features extracted from the prototype to specific path.

        Args:
            path (str): Path to save feature.
        """
        if not self.prototype_inited:
            self.prepare_prototype()
        torch.save(self.prototype_vecs, path)
