# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone


class BasicConv2d(BaseModule):
    """A basic convolution block including convolution, batch norm and ReLU.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict, optional): The config of convolution layer.
            Defaults to None, which means to use ``nn.Conv2d``.
        init_cfg (dict, optional): The config of initialization.
            Defaults to None.
        **kwargs: Other keyword arguments of the convolution layer.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.conv = build_conv_layer(
            conv_cfg, in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionA(BaseModule):
    """Type-A Inception block.

    Args:
        in_channels (int): The number of input channels.
        pool_features (int): The number of channels in pooling branch.
        conv_cfg (dict, optional): The convolution layer config in the
            :class:`BasicConv2d` block. Defaults to None.
        init_cfg (dict, optional): The config of initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 pool_features: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.branch1x1 = BasicConv2d(
            in_channels, 64, kernel_size=1, conv_cfg=conv_cfg)

        self.branch5x5_1 = BasicConv2d(
            in_channels, 48, kernel_size=1, conv_cfg=conv_cfg)
        self.branch5x5_2 = BasicConv2d(
            48, 64, kernel_size=5, padding=2, conv_cfg=conv_cfg)

        self.branch3x3dbl_1 = BasicConv2d(
            in_channels, 64, kernel_size=1, conv_cfg=conv_cfg)
        self.branch3x3dbl_2 = BasicConv2d(
            64, 96, kernel_size=3, padding=1, conv_cfg=conv_cfg)
        self.branch3x3dbl_3 = BasicConv2d(
            96, 96, kernel_size=3, padding=1, conv_cfg=conv_cfg)

        self.branch_pool_downsample = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(
            in_channels, pool_features, kernel_size=1, conv_cfg=conv_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool_downsample(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(BaseModule):
    """Type-B Inception block.

    Args:
        in_channels (int): The number of input channels.
        conv_cfg (dict, optional): The convolution layer config in the
            :class:`BasicConv2d` block. Defaults to None.
        init_cfg (dict, optional): The config of initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.branch3x3 = BasicConv2d(
            in_channels, 384, kernel_size=3, stride=2, conv_cfg=conv_cfg)

        self.branch3x3dbl_1 = BasicConv2d(
            in_channels, 64, kernel_size=1, conv_cfg=conv_cfg)
        self.branch3x3dbl_2 = BasicConv2d(
            64, 96, kernel_size=3, padding=1, conv_cfg=conv_cfg)
        self.branch3x3dbl_3 = BasicConv2d(
            96, 96, kernel_size=3, stride=2, conv_cfg=conv_cfg)

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(BaseModule):
    """Type-C Inception block.

    Args:
        in_channels (int): The number of input channels.
        channels_7x7 (int): The number of channels in 7x7 convolution branch.
        conv_cfg (dict, optional): The convolution layer config in the
            :class:`BasicConv2d` block. Defaults to None.
        init_cfg (dict, optional): The config of initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 channels_7x7: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.branch1x1 = BasicConv2d(
            in_channels, 192, kernel_size=1, conv_cfg=conv_cfg)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(
            in_channels, c7, kernel_size=1, conv_cfg=conv_cfg)
        self.branch7x7_2 = BasicConv2d(
            c7, c7, kernel_size=(1, 7), padding=(0, 3), conv_cfg=conv_cfg)
        self.branch7x7_3 = BasicConv2d(
            c7, 192, kernel_size=(7, 1), padding=(3, 0), conv_cfg=conv_cfg)

        self.branch7x7dbl_1 = BasicConv2d(
            in_channels, c7, kernel_size=1, conv_cfg=conv_cfg)
        self.branch7x7dbl_2 = BasicConv2d(
            c7, c7, kernel_size=(7, 1), padding=(3, 0), conv_cfg=conv_cfg)
        self.branch7x7dbl_3 = BasicConv2d(
            c7, c7, kernel_size=(1, 7), padding=(0, 3), conv_cfg=conv_cfg)
        self.branch7x7dbl_4 = BasicConv2d(
            c7, c7, kernel_size=(7, 1), padding=(3, 0), conv_cfg=conv_cfg)
        self.branch7x7dbl_5 = BasicConv2d(
            c7, 192, kernel_size=(1, 7), padding=(0, 3), conv_cfg=conv_cfg)

        self.branch_pool_downsample = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(
            in_channels, 192, kernel_size=1, conv_cfg=conv_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool_downsample(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(BaseModule):
    """Type-D Inception block.

    Args:
        in_channels (int): The number of input channels.
        conv_cfg (dict, optional): The convolution layer config in the
            :class:`BasicConv2d` block. Defaults to None.
        init_cfg (dict, optional): The config of initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        self.branch3x3_1 = BasicConv2d(
            in_channels, 192, kernel_size=1, conv_cfg=conv_cfg)
        self.branch3x3_2 = BasicConv2d(
            192, 320, kernel_size=3, stride=2, conv_cfg=conv_cfg)

        self.branch7x7x3_1 = BasicConv2d(
            in_channels, 192, kernel_size=1, conv_cfg=conv_cfg)
        self.branch7x7x3_2 = BasicConv2d(
            192, 192, kernel_size=(1, 7), padding=(0, 3), conv_cfg=conv_cfg)
        self.branch7x7x3_3 = BasicConv2d(
            192, 192, kernel_size=(7, 1), padding=(3, 0), conv_cfg=conv_cfg)
        self.branch7x7x3_4 = BasicConv2d(
            192, 192, kernel_size=3, stride=2, conv_cfg=conv_cfg)

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(x)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(BaseModule):
    """Type-E Inception block.

    Args:
        in_channels (int): The number of input channels.
        conv_cfg (dict, optional): The convolution layer config in the
            :class:`BasicConv2d` block. Defaults to None.
        init_cfg (dict, optional): The config of initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.branch1x1 = BasicConv2d(
            in_channels, 320, kernel_size=1, conv_cfg=conv_cfg)

        self.branch3x3_1 = BasicConv2d(
            in_channels, 384, kernel_size=1, conv_cfg=conv_cfg)
        self.branch3x3_2a = BasicConv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1), conv_cfg=conv_cfg)
        self.branch3x3_2b = BasicConv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0), conv_cfg=conv_cfg)

        self.branch3x3dbl_1 = BasicConv2d(
            in_channels, 448, kernel_size=1, conv_cfg=conv_cfg)
        self.branch3x3dbl_2 = BasicConv2d(
            448, 384, kernel_size=3, padding=1, conv_cfg=conv_cfg)
        self.branch3x3dbl_3a = BasicConv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1), conv_cfg=conv_cfg)
        self.branch3x3dbl_3b = BasicConv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0), conv_cfg=conv_cfg)

        self.branch_pool_downsample = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.branch_pool = BasicConv2d(
            in_channels, 192, kernel_size=1, conv_cfg=conv_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.branch_pool_downsample(x)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(BaseModule):
    """The Inception block for the auxiliary classification branch.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of categroies.
        conv_cfg (dict, optional): The convolution layer config in the
            :class:`BasicConv2d` block. Defaults to None.
        init_cfg (dict, optional): The config of initialization.
            Defaults to use trunc normal with ``std=0.01`` for Conv2d layers
            and use trunc normal with ``std=0.001`` for Linear layers..
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 conv_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = [
                     dict(type='TruncNormal', layer='Conv2d', std=0.01),
                     dict(type='TruncNormal', layer='Linear', std=0.001)
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.downsample = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv0 = BasicConv2d(
            in_channels, 128, kernel_size=1, conv_cfg=conv_cfg)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5, conv_cfg=conv_cfg)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # N x 768 x 17 x 17
        x = self.downsample(x)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = self.gap(x)
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


@MODELS.register_module()
class InceptionV3(BaseBackbone):
    """Inception V3 backbone.

    A PyTorch implementation of `Rethinking the Inception Architecture for
    Computer Vision <https://arxiv.org/abs/1512.00567>`_

    This implementation is modified from
    https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py.
    Licensed under the BSD 3-Clause License.

    Args:
        num_classes (int): The number of categroies. Defaults to 1000.
        aux_logits (bool): Whether to enable the auxiliary branch. If False,
            the auxiliary logits output will be None. Defaults to False.
        dropout (float): Dropout rate. Defaults to 0.5.
        init_cfg (dict, optional): The config of initialization. Defaults
            to use trunc normal with ``std=0.1`` for all Conv2d and Linear
            layers and constant with ``val=1`` for all BatchNorm2d layers.

    Example:
        >>> import torch
        >>> from mmpretrain.models import build_backbone
        >>>
        >>> inputs = torch.rand(2, 3, 299, 299)
        >>> cfg = dict(type='InceptionV3', num_classes=100)
        >>> backbone = build_backbone(cfg)
        >>> aux_out, out = backbone(inputs)
        >>> # The auxiliary branch is disabled by default.
        >>> assert aux_out is None
        >>> print(out.shape)
        torch.Size([2, 100])
        >>> cfg = dict(type='InceptionV3', num_classes=100, aux_logits=True)
        >>> backbone = build_backbone(cfg)
        >>> aux_out, out = backbone(inputs)
        >>> print(aux_out.shape, out.shape)
        torch.Size([2, 100]) torch.Size([2, 100])
    """

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = False,
        dropout: float = 0.5,
        init_cfg: Optional[dict] = [
            dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.1),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ],
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.aux_logits = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

    def forward(
            self,
            x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward function."""
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[torch.Tensor] = None
        if self.aux_logits and self.training:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return aux, x
