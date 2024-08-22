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

OptIntSeq = Optional[Sequence[int]]

@MODELS.register_module()
class VitHDRIHead(BaseDecodeHead):
    def __init__(self,
            deconv_out_channels: OptIntSeq = None,
            deconv_kernel_sizes: OptIntSeq = None,
            conv_out_channels: OptIntSeq = (256, 256, 256),
            conv_kernel_sizes: OptIntSeq = (4, 4, 4),
            final_layer: OptIntSeq = (1536, 1536, 1536),
            hdri_size: Tuple[int, int] = (16, 32), ## height and width
            interpolate_mode='bilinear',
            **kwargs):
        super().__init__(init_cfg=dict(), **kwargs)

        self.interpolate_mode = interpolate_mode
        self.hdri_size = hdri_size

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

        self.final_layer = self._make_final_layer(final_layer)

        return

    def _make_final_layer(self, final_layer: Sequence[int]) -> nn.Module:
        """Create final layer by given parameters."""

        layers = [nn.Flatten()]
        in_features = final_layer[0]

        for i in range(1, len(final_layer)):
            layers.append(nn.Linear(in_features, final_layer[i]))
            if i < len(final_layer) - 1:  # No activation after the last layer
                layers.append(nn.SiLU())
            in_features = final_layer[i]

        return nn.Sequential(*layers)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            stride = 2 # Set stride to 2 to reduce resolution by half
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
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

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs) ## inputs is B x 1536 x 64 x 48

        x = self.deconv_layers(inputs) ## no change
        x = self.conv_layers(x) ## x is B x 768 x 8 x 6

        out = self.final_layer(x) ## B x 1536
        out = out.view(out.size(0), 3, self.hdri_size[0], self.hdri_size[1]) ## B x 3 x 16 x 32. B x C x H x W

        return out

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses, seg_logits

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_depth_maps = [data_sample.gt_depth_map.data for data_sample in batch_data_samples]
        return torch.stack(gt_depth_maps, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples) ## B x 3 x 16 x 32

        loss = dict()
        seg_weight = None

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,)

        return loss

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)

        return seg_logits
