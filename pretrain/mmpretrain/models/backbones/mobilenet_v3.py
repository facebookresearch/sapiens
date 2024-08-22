# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.registry import MODELS
from ..utils import InvertedResidual
from .base_backbone import BaseBackbone


@MODELS.register_module()
class MobileNetV3(BaseBackbone):
    """MobileNetV3 backbone.

    Args:
        arch (str): Architecture of mobilnetv3, from {small, large}.
            Default: small.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (None or Sequence[int]): Output from which stages.
            Default: None, which means output tensors from final stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride]
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2],
                  [3, 72, 24, False, 'ReLU', 2],
                  [3, 88, 24, False, 'ReLU', 1],
                  [5, 96, 40, True, 'HSwish', 2],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 120, 48, True, 'HSwish', 1],
                  [5, 144, 48, True, 'HSwish', 1],
                  [5, 288, 96, True, 'HSwish', 2],
                  [5, 576, 96, True, 'HSwish', 1],
                  [5, 576, 96, True, 'HSwish', 1]],
        'small_075': [[3, 16, 16, True, 'ReLU', 2],
                      [3, 72, 24, False, 'ReLU', 2],
                      [3, 88, 24, False, 'ReLU', 1],
                      [5, 96, 32, True, 'HSwish', 2],
                      [5, 192, 32, True, 'HSwish', 1],
                      [5, 192, 32, True, 'HSwish', 1],
                      [5, 96, 40, True, 'HSwish', 1],
                      [5, 120, 40, True, 'HSwish', 1],
                      [5, 240, 72, True, 'HSwish', 2],
                      [5, 432, 72, True, 'HSwish', 1],
                      [5, 432, 72, True, 'HSwish', 1]],
        'small_050': [[3, 16, 8, True, 'ReLU', 2],
                      [3, 40, 16, False, 'ReLU', 2],
                      [3, 56, 16, False, 'ReLU', 1],
                      [5, 64, 24, True, 'HSwish', 2],
                      [5, 144, 24, True, 'HSwish', 1],
                      [5, 144, 24, True, 'HSwish', 1],
                      [5, 72, 24, True, 'HSwish', 1],
                      [5, 72, 24, True, 'HSwish', 1],
                      [5, 144, 48, True, 'HSwish', 2],
                      [5, 288, 48, True, 'HSwish', 1],
                      [5, 288, 48, True, 'HSwish', 1]],
        'large': [[3, 16, 16, False, 'ReLU', 1],
                  [3, 64, 24, False, 'ReLU', 2],
                  [3, 72, 24, False, 'ReLU', 1],
                  [5, 72, 40, True, 'ReLU', 2],
                  [5, 120, 40, True, 'ReLU', 1],
                  [5, 120, 40, True, 'ReLU', 1],
                  [3, 240, 80, False, 'HSwish', 2],
                  [3, 200, 80, False, 'HSwish', 1],
                  [3, 184, 80, False, 'HSwish', 1],
                  [3, 184, 80, False, 'HSwish', 1],
                  [3, 480, 112, True, 'HSwish', 1],
                  [3, 672, 112, True, 'HSwish', 1],
                  [5, 672, 160, True, 'HSwish', 2],
                  [5, 960, 160, True, 'HSwish', 1],
                  [5, 960, 160, True, 'HSwish', 1]]
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                 out_indices=None,
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='Kaiming',
                         layer=['Conv2d'],
                         nonlinearity='leaky_relu'),
                     dict(type='Normal', layer=['Linear'], std=0.01),
                     dict(type='Constant', layer=['BatchNorm2d'], val=1)
                 ]):
        super(MobileNetV3, self).__init__(init_cfg)
        assert arch in self.arch_settings
        if out_indices is None:
            out_indices = (12, ) if 'small' in arch else (16, )
        for order, index in enumerate(out_indices):
            if index not in range(0, len(self.arch_settings[arch]) + 2):
                raise ValueError(
                    'the item in out_indices must in '
                    f'range(0, {len(self.arch_settings[arch]) + 2}). '
                    f'But received {index}')

        if frozen_stages not in range(-1, len(self.arch_settings[arch]) + 2):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{len(self.arch_settings[arch]) + 2}). '
                             f'But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layers = self._make_layer()
        self.feat_dim = self.arch_settings[arch][-1][1]

    def _make_layer(self):
        layers = []
        layer_setting = self.arch_settings[self.arch]
        in_channels = 16

        layer = ConvModule(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='HSwish'))
        self.add_module('layer0', layer)
        layers.append('layer0')

        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act,
             stride) = params
            if with_se:
                se_cfg = dict(
                    channels=mid_channels,
                    ratio=4,
                    act_cfg=(dict(type='ReLU'),
                             dict(
                                 type='HSigmoid',
                                 bias=3,
                                 divisor=6,
                                 min_value=0,
                                 max_value=1)))
            else:
                se_cfg = None

            layer = InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                se_cfg=se_cfg,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type=act),
                with_cp=self.with_cp)
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)

        # Build the last layer before pooling
        # TODO: No dilation
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='HSwish'))
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)

        return layers

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
