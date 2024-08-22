# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import build_conv_layer, build_upsample_layer

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from typing import List, Tuple, Optional, Sequence, Union
from torch import Tensor
from mmseg.utils import ConfigType, SampleList
import torch.nn.functional as F

OptIntSeq = Optional[Sequence[int]]

@MODELS.register_module()
class VitStereoCorrespondencesHead(BaseDecodeHead):
    def __init__(self,
            deconv_out_channels: OptIntSeq = None,
            deconv_kernel_sizes: OptIntSeq = None,
            upsample_conv_out_channels: OptIntSeq = None,
            upsample_conv_kernel_sizes: OptIntSeq = None,
            conv_out_channels: OptIntSeq = None,
            conv_kernel_sizes: OptIntSeq = None,
            final_layer: dict = dict(kernel_size=1),
            interpolate_mode='bilinear', **kwargs):
        super().__init__(**kwargs)

        self.interpolate_mode = interpolate_mode

        in_channels = self.in_channels

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if upsample_conv_out_channels:
            if upsample_conv_kernel_sizes is None or len(upsample_conv_out_channels) != len(
                    upsample_conv_kernel_sizes):
                raise ValueError(
                    '"upsample_conv_out_channels" and "upsample_conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {upsample_conv_out_channels} and '
                    f'{upsample_conv_kernel_sizes}')

            self.upsample_conv_layers = self._make_upsample_conv_layers(
                in_channels=in_channels,
                layer_out_channels=upsample_conv_out_channels,
                layer_kernel_sizes=upsample_conv_kernel_sizes)
            in_channels = upsample_conv_out_channels[-1]
        else:
            self.upsample_conv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        return

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_upsample_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)

            ### append a 2x upsampling layer, F.interpolate(..., mode='bicubic')
            layers.append(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=self.align_corners))
            layers.append(build_conv_layer(cfg))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, inputs, normalize=True):
        inputs = self._transform_inputs(inputs)
        x = self.deconv_layers(inputs)
        x = self.upsample_conv_layers(x)
        x = self.conv_layers(x)
        out = self.conv_seg(x) ## B x D x H x W, 1 x 32 x 512 x 384

        ## normalize to unit norm per pixel
        if normalize: 
            out = F.normalize(out, p=2, dim=1)
        
        return out

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        descs = self.forward(inputs)
        losses = self.loss_by_feat(descs, batch_data_samples)
        return losses, descs

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        return loss

    def predict_by_feat(self, descs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        descs = F.interpolate(descs, size=batch_img_metas[0]['img_shape'], mode='bilinear', align_corners=self.align_corners)
        return descs
