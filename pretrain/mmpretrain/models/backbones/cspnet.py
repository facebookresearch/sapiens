# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.registry import MODELS
from ..utils import to_ntuple
from .resnet import Bottleneck as ResNetBottleneck
from .resnext import Bottleneck as ResNeXtBottleneck

eps = 1.0e-5


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet. Each DarknetBottleneck
    consists of two ConvModules and the input is added to the final output.
    Each ConvModule is composed of Conv, BN, and LeakyReLU. The first convLayer
    has filter size of 1x1 and the second one has the filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2.
            Defaults to 4.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        drop_path_rate (float): The ratio of the drop path layer. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN', eps=1e-5)``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='Swish')``.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=2,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 drop_path_rate=0,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels / expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop_path(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPStage(BaseModule):
    """Cross Stage Partial Stage.

    .. code:: text

        Downsample Convolution (optional)
                    |
                    |
            Expand Convolution
                    |
                    |
           Split to xa, xb
                    |     \
                    |      \
                    |      blocks(xb)
                    |      /
                    |     /  transition
                    |    /
            Concat xa, blocks(xb)
                    |
         Transition Convolution

    Args:
        block_fn (nn.module): The basic block function in the Stage.
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        has_downsampler (bool): Whether to add a downsampler in the stage.
            Default: False.
        down_growth (bool): Whether to expand the channels in the
            downsampler layer of the stage. Default: False.
        expand_ratio (float): The expand ratio to adjust the number of
             channels of the expand conv layer. Default: 0.5
        bottle_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        block_dpr (float): The ratio of the drop path layer in the
            blocks of the stage. Default: 0.
        num_blocks (int): Number of blocks. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', inplace=True)
    """

    def __init__(self,
                 block_fn,
                 in_channels,
                 out_channels,
                 has_downsampler=True,
                 down_growth=False,
                 expand_ratio=0.5,
                 bottle_ratio=2,
                 num_blocks=1,
                 block_dpr=0,
                 block_args={},
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        # grow downsample channels to output channels
        down_channels = out_channels if down_growth else in_channels
        block_dpr = to_ntuple(num_blocks)(block_dpr)

        if has_downsampler:
            self.downsample_conv = ConvModule(
                in_channels=in_channels,
                out_channels=down_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=32 if block_fn is ResNeXtBottleneck else 1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.downsample_conv = nn.Identity()

        exp_channels = int(down_channels * expand_ratio)
        self.expand_conv = ConvModule(
            in_channels=down_channels,
            out_channels=exp_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg if block_fn is DarknetBottleneck else None)

        assert exp_channels % 2 == 0, \
            'The channel number before blocks must be divisible by 2.'
        block_channels = exp_channels // 2
        blocks = []
        for i in range(num_blocks):
            block_cfg = dict(
                in_channels=block_channels,
                out_channels=block_channels,
                expansion=bottle_ratio,
                drop_path_rate=block_dpr[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **block_args)
            blocks.append(block_fn(**block_cfg))
        self.blocks = Sequential(*blocks)
        self.atfer_blocks_conv = ConvModule(
            block_channels,
            block_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.final_conv = ConvModule(
            2 * block_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.downsample_conv(x)
        x = self.expand_conv(x)

        split = x.shape[1] // 2
        xa, xb = x[:, :split], x[:, split:]

        xb = self.blocks(xb)
        xb = self.atfer_blocks_conv(xb).contiguous()

        x_final = torch.cat((xa, xb), dim=1)
        return self.final_conv(x_final)


class CSPNet(BaseModule):
    """The abstract CSP Network class.

    A Pytorch implementation of `CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN <https://arxiv.org/abs/1911.11929>`_

    This class is an abstract class because the Cross Stage Partial Network
    (CSPNet) is a kind of universal network structure, and you
    network block to implement networks like CSPResNet, CSPResNeXt and
    CSPDarkNet.

    Args:
        arch (dict): The architecture of the CSPNet.
            It should have the following keys:

            - block_fn (Callable): A function or class to return a block
              module, and it should accept at least ``in_channels``,
              ``out_channels``, ``expansion``, ``drop_path_rate``, ``norm_cfg``
              and ``act_cfg``.
            - in_channels (Tuple[int]): The number of input channels of each
              stage.
            - out_channels (Tuple[int]): The number of output channels of each
              stage.
            - num_blocks (Tuple[int]): The number of blocks in each stage.
            - expansion_ratio (float | Tuple[float]): The expansion ratio in
              the expand convolution of each stage. Defaults to 0.5.
            - bottle_ratio (float | Tuple[float]): The expansion ratio of
              blocks in each stage. Defaults to 2.
            - has_downsampler (bool | Tuple[bool]): Whether to add a
              downsample convolution in each stage. Defaults to True
            - down_growth (bool | Tuple[bool]): Whether to expand the channels
              in the downsampler layer of each stage. Defaults to False.
            - block_args (dict | Tuple[dict], optional): The extra arguments to
              the blocks in each stage. Defaults to None.

        stem_fn (Callable): A function or class to return a stem module.
            And it should accept ``in_channels``.
        in_channels (int): Number of input image channels. Defaults to 3.
        out_indices (int | Sequence[int]): Output from which stages.
            Defaults to -1, which means the last stage.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        conv_cfg (dict, optional): The config dict for conv layers in blocks.
            Defaults to None, which means use Conv2d.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN', eps=1e-5)``.
        act_cfg (dict): The config dict for activation functions.
            Defaults to ``dict(type='LeakyReLU', inplace=True)``.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict, optional): The initialization settings.
            Defaults to ``dict(type='Kaiming', layer='Conv2d'))``.

    Example:
        >>> from functools import partial
        >>> import torch
        >>> import torch.nn as nn
        >>> from mmpretrain.models import CSPNet
        >>> from mmpretrain.models.backbones.resnet import Bottleneck
        >>>
        >>> # A simple example to build CSPNet.
        >>> arch = dict(
        ...     block_fn=Bottleneck,
        ...     in_channels=[32, 64],
        ...     out_channels=[64, 128],
        ...     num_blocks=[3, 4]
        ... )
        >>> stem_fn = partial(nn.Conv2d, out_channels=32, kernel_size=3)
        >>> model = CSPNet(arch=arch, stem_fn=stem_fn, out_indices=(0, 1))
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> outs = model(inputs)
        >>> for out in outs:
        ...     print(out.shape)
        ...
        (1, 64, 111, 111)
        (1, 128, 56, 56)
    """

    def __init__(self,
                 arch,
                 stem_fn,
                 in_channels=3,
                 out_indices=-1,
                 frozen_stages=-1,
                 drop_path_rate=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 norm_eval=False,
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg=init_cfg)
        self.arch = self.expand_arch(arch)
        self.num_stages = len(self.arch['in_channels'])
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        if frozen_stages not in range(-1, self.num_stages):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{self.num_stages}). But received '
                             f'{frozen_stages}')
        self.frozen_stages = frozen_stages

        self.stem = stem_fn(in_channels)

        stages = []
        depths = self.arch['num_blocks']
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).split(depths)

        for i in range(self.num_stages):
            stage_cfg = {k: v[i] for k, v in self.arch.items()}
            csp_stage = CSPStage(
                **stage_cfg,
                block_dpr=dpr[i].tolist(),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                init_cfg=init_cfg)
            stages.append(csp_stage)
        self.stages = Sequential(*stages)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.stages) + index
            assert 0 <= out_indices[i] <= len(self.stages), \
                f'Invalid out_indices {index}.'
        self.out_indices = out_indices

    @staticmethod
    def expand_arch(arch):
        num_stages = len(arch['in_channels'])

        def to_tuple(x, name=''):
            if isinstance(x, (list, tuple)):
                assert len(x) == num_stages, \
                    f'The length of {name} ({len(x)}) does not ' \
                    f'equals to the number of stages ({num_stages})'
                return tuple(x)
            else:
                return (x, ) * num_stages

        full_arch = {k: to_tuple(v, k) for k, v in arch.items()}
        if 'block_args' not in full_arch:
            full_arch['block_args'] = to_tuple({})
        return full_arch

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(CSPNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []

        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class CSPDarkNet(CSPNet):
    """CSP-Darknet backbone used in YOLOv4.

    Args:
        depth (int): Depth of CSP-Darknet. Default: 53.
        in_channels (int): Number of input image channels. Default: 3.
        out_indices (Sequence[int]): Output from which stages.
            Default: (3, ).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmpretrain.models import CSPDarkNet
        >>> import torch
        >>> model = CSPDarkNet(depth=53, out_indices=(0, 1, 2, 3, 4))
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 64, 208, 208)
        (1, 128, 104, 104)
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    arch_settings = {
        53:
        dict(
            block_fn=DarknetBottleneck,
            in_channels=(32, 64, 128, 256, 512),
            out_channels=(64, 128, 256, 512, 1024),
            num_blocks=(1, 2, 8, 8, 4),
            expand_ratio=(2, 1, 1, 1, 1),
            bottle_ratio=(2, 1, 1, 1, 1),
            has_downsampler=True,
            down_growth=True,
        ),
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 out_indices=(4, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):

        assert depth in self.arch_settings, 'depth must be one of ' \
            f'{list(self.arch_settings.keys())}, but get {depth}.'

        super().__init__(
            arch=self.arch_settings[depth],
            stem_fn=self._make_stem_layer,
            in_channels=in_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def _make_stem_layer(self, in_channels):
        """using a stride=1 conv as the stem in CSPDarknet."""
        # `stem_channels` equals to the `in_channels` in the first stage.
        stem_channels = self.arch['in_channels'][0]
        stem = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        return stem


@MODELS.register_module()
class CSPResNet(CSPNet):
    """CSP-ResNet backbone.

    Args:
        depth (int): Depth of CSP-ResNet. Default: 50.
        out_indices (Sequence[int]): Output from which stages.
            Default: (4, ).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmpretrain.models import CSPResNet
        >>> import torch
        >>> model = CSPResNet(depth=50, out_indices=(0, 1, 2, 3))
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 128, 104, 104)
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    arch_settings = {
        50:
        dict(
            block_fn=ResNetBottleneck,
            in_channels=(64, 128, 256, 512),
            out_channels=(128, 256, 512, 1024),
            num_blocks=(3, 3, 5, 2),
            expand_ratio=4,
            bottle_ratio=2,
            has_downsampler=(False, True, True, True),
            down_growth=False),
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 deep_stem=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-5),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 norm_eval=False,
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        assert depth in self.arch_settings, 'depth must be one of ' \
            f'{list(self.arch_settings.keys())}, but get {depth}.'
        self.deep_stem = deep_stem

        super().__init__(
            arch=self.arch_settings[depth],
            stem_fn=self._make_stem_layer,
            in_channels=in_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def _make_stem_layer(self, in_channels):
        # `stem_channels` equals to the `in_channels` in the first stage.
        stem_channels = self.arch['in_channels'][0]
        if self.deep_stem:
            stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        else:
            stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return stem


@MODELS.register_module()
class CSPResNeXt(CSPResNet):
    """CSP-ResNeXt backbone.

    Args:
        depth (int): Depth of CSP-ResNeXt. Default: 50.
        out_indices (Sequence[int]): Output from which stages.
            Default: (4, ).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmpretrain.models import CSPResNeXt
        >>> import torch
        >>> model = CSPResNeXt(depth=50, out_indices=(0, 1, 2, 3))
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 56, 56)
        (1, 512, 28, 28)
        (1, 1024, 14, 14)
        (1, 2048, 7, 7)
    """
    arch_settings = {
        50:
        dict(
            block_fn=ResNeXtBottleneck,
            in_channels=(64, 256, 512, 1024),
            out_channels=(256, 512, 1024, 2048),
            num_blocks=(3, 3, 5, 2),
            expand_ratio=(4, 2, 2, 2),
            bottle_ratio=4,
            has_downsampler=(False, True, True, True),
            down_growth=False,
            # the base_channels is changed from 64 to 32 in CSPNet
            block_args=dict(base_channels=32),
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
