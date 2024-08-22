# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule, DropPath
from mmengine.model import Sequential
from torch import Tensor

from mmpretrain.registry import MODELS
from ..utils import InvertedResidual as MBConv
from .base_backbone import BaseBackbone
from .efficientnet import EdgeResidual as FusedMBConv


class EnhancedConvModule(ConvModule):
    """ConvModule with short-cut and droppath.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        has_skip (bool): Whether there is short-cut. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    def __init__(self, *args, has_skip=False, drop_path_rate=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_skip = has_skip
        if self.has_skip and (self.in_channels != self.out_channels
                              or self.stride != (1, 1)):
            raise ValueError('the stride must be 1 and the `in_channels` and'
                             ' `out_channels` must be the same , when '
                             '`has_skip` is True in `EnhancedConvModule` .')
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        short_cut = x
        x = super().forward(x, **kwargs)
        if self.has_skip:
            x = self.drop_path(x) + short_cut
        return x


@MODELS.register_module()
class EfficientNetV2(BaseBackbone):
    """EfficientNetV2 backbone.

    A PyTorch implementation of EfficientNetV2 introduced by:
    `EfficientNetV2: Smaller Models and Faster Training
    <https://arxiv.org/abs/2104.00298>`_

    Args:
        arch (str): Architecture of efficientnetv2. Defaults to s.
        in_channels (int): Number of input image channels. Defaults to 3.
        drop_path_rate (float): The ratio of the stochastic depth.
            Defaults to 0.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (-1, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """

    # Parameters to build layers. From left to right:
    # - repeat (int): The repeat number of the block in the layer
    # - kernel_size (int): The kernel size of the layer
    # - stride (int): The stride of the first block of the layer
    # - expand_ratio (int, float): The expand_ratio of the mid_channels
    # - in_channel (int): The number of in_channels of the layer
    # - out_channel (int): The number of out_channels of the layer
    # - se_ratio (float): The sequeeze ratio of SELayer.
    # - block_type (int): -2: ConvModule, -1: EnhancedConvModule,
    #                      0: FusedMBConv, 1: MBConv
    arch_settings = {
        **dict.fromkeys(['small', 's'], [[2, 3, 1, 1, 24, 24, 0.0, -1],
                                         [4, 3, 2, 4, 24, 48, 0.0, 0],
                                         [4, 3, 2, 4, 48, 64, 0.0, 0],
                                         [6, 3, 2, 4, 64, 128, 0.25, 1],
                                         [9, 3, 1, 6, 128, 160, 0.25, 1],
                                         [15, 3, 2, 6, 160, 256, 0.25, 1],
                                         [1, 1, 1, 1, 256, 1280, 0.0, -2]]),
        **dict.fromkeys(['m', 'medium'], [[3, 3, 1, 1, 24, 24, 0.0, -1],
                                          [5, 3, 2, 4, 24, 48, 0.0, 0],
                                          [5, 3, 2, 4, 48, 80, 0.0, 0],
                                          [7, 3, 2, 4, 80, 160, 0.25, 1],
                                          [14, 3, 1, 6, 160, 176, 0.25, 1],
                                          [18, 3, 2, 6, 176, 304, 0.25, 1],
                                          [5, 3, 1, 6, 304, 512, 0.25, 1],
                                          [1, 1, 1, 1, 512, 1280, 0.0, -2]]),
        **dict.fromkeys(['l', 'large'], [[4, 3, 1, 1, 32, 32, 0.0, -1],
                                         [7, 3, 2, 4, 32, 64, 0.0, 0],
                                         [7, 3, 2, 4, 64, 96, 0.0, 0],
                                         [10, 3, 2, 4, 96, 192, 0.25, 1],
                                         [19, 3, 1, 6, 192, 224, 0.25, 1],
                                         [25, 3, 2, 6, 224, 384, 0.25, 1],
                                         [7, 3, 1, 6, 384, 640, 0.25, 1],
                                         [1, 1, 1, 1, 640, 1280, 0.0, -2]]),
        **dict.fromkeys(['xl'], [[4, 3, 1, 1, 32, 32, 0.0, -1],
                                 [8, 3, 2, 4, 32, 64, 0.0, 0],
                                 [8, 3, 2, 4, 64, 96, 0.0, 0],
                                 [16, 3, 2, 4, 96, 192, 0.25, 1],
                                 [24, 3, 1, 6, 192, 256, 0.25, 1],
                                 [32, 3, 2, 6, 256, 512, 0.25, 1],
                                 [8, 3, 1, 6, 512, 640, 0.25, 1],
                                 [1, 1, 1, 1, 640, 1280, 0.0, -2]]),
        **dict.fromkeys(['b0'], [[1, 3, 1, 1, 32, 16, 0.0, -1],
                                 [2, 3, 2, 4, 16, 32, 0.0, 0],
                                 [2, 3, 2, 4, 32, 48, 0.0, 0],
                                 [3, 3, 2, 4, 48, 96, 0.25, 1],
                                 [5, 3, 1, 6, 96, 112, 0.25, 1],
                                 [8, 3, 2, 6, 112, 192, 0.25, 1],
                                 [1, 1, 1, 1, 192, 1280, 0.0, -2]]),
        **dict.fromkeys(['b1'], [[2, 3, 1, 1, 32, 16, 0.0, -1],
                                 [3, 3, 2, 4, 16, 32, 0.0, 0],
                                 [3, 3, 2, 4, 32, 48, 0.0, 0],
                                 [4, 3, 2, 4, 48, 96, 0.25, 1],
                                 [6, 3, 1, 6, 96, 112, 0.25, 1],
                                 [9, 3, 2, 6, 112, 192, 0.25, 1],
                                 [1, 1, 1, 1, 192, 1280, 0.0, -2]]),
        **dict.fromkeys(['b2'], [[2, 3, 1, 1, 32, 16, 0.0, -1],
                                 [3, 3, 2, 4, 16, 32, 0.0, 0],
                                 [3, 3, 2, 4, 32, 56, 0.0, 0],
                                 [4, 3, 2, 4, 56, 104, 0.25, 1],
                                 [6, 3, 1, 6, 104, 120, 0.25, 1],
                                 [10, 3, 2, 6, 120, 208, 0.25, 1],
                                 [1, 1, 1, 1, 208, 1408, 0.0, -2]]),
        **dict.fromkeys(['b3'], [[2, 3, 1, 1, 40, 16, 0.0, -1],
                                 [3, 3, 2, 4, 16, 40, 0.0, 0],
                                 [3, 3, 2, 4, 40, 56, 0.0, 0],
                                 [5, 3, 2, 4, 56, 112, 0.25, 1],
                                 [7, 3, 1, 6, 112, 136, 0.25, 1],
                                 [12, 3, 2, 6, 136, 232, 0.25, 1],
                                 [1, 1, 1, 1, 232, 1536, 0.0, -2]])
    }

    def __init__(self,
                 arch: str = 's',
                 in_channels: int = 3,
                 drop_path_rate: float = 0.,
                 out_indices: Sequence[int] = (-1, ),
                 frozen_stages: int = 0,
                 conv_cfg=dict(type='Conv2dAdaptivePadding'),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.1),
                 act_cfg=dict(type='Swish'),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(
                         type='Constant',
                         layer=['_BatchNorm', 'GroupNorm'],
                         val=1)
                 ]):
        super(EfficientNetV2, self).__init__(init_cfg)
        assert arch in self.arch_settings, \
            f'"{arch}" is not one of the arch_settings ' \
            f'({", ".join(self.arch_settings.keys())})'
        self.arch = self.arch_settings[arch]
        if frozen_stages not in range(len(self.arch) + 1):
            raise ValueError('frozen_stages must be in range(0, '
                             f'{len(self.arch)}), but get {frozen_stages}')
        self.drop_path_rate = drop_path_rate
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layers = nn.ModuleList()
        assert self.arch[-1][-1] == -2, \
            f'the last block_type of `arch_setting` must be -2 ,' \
            f'but get `{self.arch[-1][-1]}`'
        self.in_channels = in_channels
        self.out_channels = self.arch[-1][5]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.make_layers()

        # there len(slef.arch) + 2 layers in the backbone
        # including: the first + len(self.arch) layers + the last
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.layers) + index
            assert 0 <= out_indices[i] <= len(self.layers), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

    def make_layers(self, ):
        # make the first layer
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.arch[0][4],
                kernel_size=3,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        in_channels = self.arch[0][4]
        layer_setting = self.arch[:-1]

        total_num_blocks = sum([x[0] for x in layer_setting])
        block_idx = 0
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, total_num_blocks)
        ]  # stochastic depth decay rule

        for layer_cfg in layer_setting:
            layer = []
            (repeat, kernel_size, stride, expand_ratio, _, out_channels,
             se_ratio, block_type) = layer_cfg
            for i in range(repeat):
                stride = stride if i == 0 else 1
                if block_type == -1:
                    has_skip = stride == 1 and in_channels == out_channels
                    droppath_rate = dpr[block_idx] if has_skip else 0.0
                    layer.append(
                        EnhancedConvModule(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            has_skip=has_skip,
                            drop_path_rate=droppath_rate,
                            stride=stride,
                            padding=1,
                            conv_cfg=None,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    in_channels = out_channels
                else:
                    mid_channels = int(in_channels * expand_ratio)
                    se_cfg = None
                    if block_type != 0 and se_ratio > 0:
                        se_cfg = dict(
                            channels=mid_channels,
                            ratio=expand_ratio * (1.0 / se_ratio),
                            divisor=1,
                            act_cfg=(self.act_cfg, dict(type='Sigmoid')))
                    block = FusedMBConv if block_type == 0 else MBConv
                    conv_cfg = self.conv_cfg if stride == 2 else None
                    layer.append(
                        block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            mid_channels=mid_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            se_cfg=se_cfg,
                            conv_cfg=conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            drop_path_rate=dpr[block_idx],
                            with_cp=self.with_cp))
                    in_channels = out_channels
                block_idx += 1
            self.layers.append(Sequential(*layer))

        # make the last layer
        self.layers.append(
            ConvModule(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=self.arch[-1][1],
                stride=self.arch[-1][2],
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(EfficientNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
