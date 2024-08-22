# Copyright (c) OpenMMLab. All right reserved.
import re
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.utils import require
from .base import BaseClassifier


@MODELS.register_module()
class HuggingFaceClassifier(BaseClassifier):
    """Image classifiers for HuggingFace model.

    This class accepts all positional and keyword arguments of the API
    ``from_pretrained`` (when ``pretrained=True``) and ``from_config`` (when
    ``pretrained=False``) of `transformers.AutoModelForImageClassification`_
    and use it to create a model from hugging-face.

    It can load checkpoints of hugging-face directly, and the saved checkpoints
    also can be directly load by hugging-face.

    Please confirm that you have installed ``transfromers`` if you want to use it.

    .. _transformers.AutoModelForImageClassification:
        https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForImageClassification

    Args:
        model_name (str): The name of the model to use in hugging-face.
        pretrained (bool): Whether to load pretrained checkpoint from
            hugging-face. Defaults to False.
        *args: Other positional arguments of the method
            `from_pretrained` or `from_config`.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in :mod:`mmpretrain.model.utils.augment`.

            Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
        **kwargs: Other keyword arguments of the method
            `from_pretrained` or `from_config`.

    Examples:
        >>> import torch
        >>> from mmpretrain.models import build_classifier
        >>> cfg = dict(type='HuggingFaceClassifier', model_name='microsoft/resnet-50', pretrained=True)
        >>> model = build_classifier(cfg)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> out = model(inputs)
        >>> print(out.shape)
        torch.Size([1, 1000])
    """  # noqa: E501

    @require('transformers')
    def __init__(self,
                 model_name,
                 pretrained=False,
                 *model_args,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 train_cfg: Optional[dict] = None,
                 with_cp: bool = False,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmpretrain.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        from transformers import AutoConfig, AutoModelForImageClassification
        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name, *model_args, **kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name, *model_args,
                                                **kwargs)
            self.model = AutoModelForImageClassification.from_config(config)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

        self.with_cp = with_cp
        if self.with_cp:
            self.model.gradient_checkpointing_enable()

        self._register_state_dict_hook(self._remove_state_dict_prefix)
        self._register_load_state_dict_pre_hook(self._add_state_dict_prefix)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'tensor':
            return self.model(inputs).logits
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        raise NotImplementedError(
            "The HuggingFaceClassifier doesn't support extract feature yet.")

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments of the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.model(inputs).logits

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None):
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.

        Returns:
            List[DataSample]: The prediction results.
        """
        # The part can be traced by torch.fx
        cls_score = self.model(inputs).logits

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

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

    @staticmethod
    def _remove_state_dict_prefix(self, state_dict, prefix, local_metadata):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = re.sub(f'^{prefix}model.', prefix, k)
            new_state_dict[new_key] = v
        return new_state_dict

    @staticmethod
    def _add_state_dict_prefix(state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        new_prefix = prefix + 'model.'
        for k in list(state_dict.keys()):
            new_key = re.sub(f'^{prefix}', new_prefix, k)
            state_dict[new_key] = state_dict[k]
            del state_dict[k]
