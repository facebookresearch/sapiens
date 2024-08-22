# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from mmengine.logging import MMLogger

from mmpretrain.registry import MODELS
from mmpretrain.utils import require
from .base_backbone import BaseBackbone


def print_timm_feature_info(feature_info):
    """Print feature_info of timm backbone to help development and debug.

    Args:
        feature_info (list[dict] | timm.models.features.FeatureInfo | None):
            feature_info of timm backbone.
    """
    logger = MMLogger.get_current_instance()
    if feature_info is None:
        logger.warning('This backbone does not have feature_info')
    elif isinstance(feature_info, list):
        for feat_idx, each_info in enumerate(feature_info):
            logger.info(f'backbone feature_info[{feat_idx}]: {each_info}')
    else:
        try:
            logger.info(f'backbone out_indices: {feature_info.out_indices}')
            logger.info(f'backbone out_channels: {feature_info.channels()}')
            logger.info(f'backbone out_strides: {feature_info.reduction()}')
        except AttributeError:
            logger.warning('Unexpected format of backbone feature_info')


@MODELS.register_module()
class TIMMBackbone(BaseBackbone):
    """Wrapper to use backbones from timm library.

    More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_.
    See especially the document for `feature extraction
    <https://rwightman.github.io/pytorch-image-models/feature_extraction/>`_.

    Args:
        model_name (str): Name of timm model to instantiate.
        features_only (bool): Whether to extract feature pyramid (multi-scale
            feature maps from the deepest layer at each stride). For Vision
            Transformer models that do not support this argument,
            set this False. Defaults to False.
        pretrained (bool): Whether to load pretrained weights.
            Defaults to False.
        checkpoint_path (str): Path of checkpoint to load at the last of
            ``timm.create_model``. Defaults to empty string, which means
            not loading.
        in_channels (int): Number of input image channels. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization config dict of
            OpenMMLab projects. Defaults to None.
        **kwargs: Other timm & model specific arguments.
    """

    @require('timm')
    def __init__(self,
                 model_name,
                 features_only=False,
                 pretrained=False,
                 checkpoint_path='',
                 in_channels=3,
                 init_cfg=None,
                 **kwargs):
        import timm

        if not isinstance(pretrained, bool):
            raise TypeError('pretrained must be bool, not str for model path')
        if features_only and checkpoint_path:
            warnings.warn(
                'Using both features_only and checkpoint_path will cause error'
                ' in timm. See '
                'https://github.com/rwightman/pytorch-image-models/issues/488')

        super(TIMMBackbone, self).__init__(init_cfg)
        if 'norm_layer' in kwargs:
            norm_class = MODELS.get(kwargs['norm_layer'])

            def build_norm(*args, **kwargs):
                return norm_class(*args, **kwargs)

            kwargs['norm_layer'] = build_norm
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs)

        # reset classifier
        if hasattr(self.timm_model, 'reset_classifier'):
            self.timm_model.reset_classifier(0, '')

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

        feature_info = getattr(self.timm_model, 'feature_info', None)
        print_timm_feature_info(feature_info)

    def forward(self, x):
        features = self.timm_model(x)
        if isinstance(features, (list, tuple)):
            features = tuple(features)
        else:
            features = (features, )
        return features
