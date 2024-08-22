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
class VitStereoPointmapHead(BaseDecodeHead):
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

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        x = self.deconv_layers(inputs)
        x = self.upsample_conv_layers(x)
        x = self.conv_layers(x)
        out = self.conv_seg(x)

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

        seg_label = self._stack_batch_gt(batch_data_samples) ## B x 1 x 1024 x 768

        device = seg_logits.device
        gt_K = [torch.from_numpy(data_sample.K).to(device) for data_sample in batch_data_samples]
        gt_K = torch.stack(gt_K, dim=0) ## B x 3 x 3

        loss = dict()

        ## -----resize for 3D, --------------
        seg_logits = F.interpolate(seg_logits, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        ## ------------------------------------------------

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None

        seg_logits = seg_logits.squeeze(1) ## B x 1024 x 768
        seg_label = seg_label.squeeze(1) ## B x 1024 x 768

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            ## pointmap consistency loss
            if loss_decode.loss_name == 'loss_consistency':
                this_loss = loss_decode(
                    seg_logits,
                    seg_label,
                    gt_K,
                    weight=seg_weight,)
            else:
                this_loss = loss_decode(
                            seg_logits,
                            seg_label,
                            weight=seg_weight,)

            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = this_loss
            else:
                loss[loss_decode.loss_name] += this_loss

        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        ## -----resize for 3D, --------------
        seg_logits = F.interpolate(seg_logits, size=batch_img_metas[0]['img_shape'], mode='bilinear', align_corners=self.align_corners)

        return seg_logits
