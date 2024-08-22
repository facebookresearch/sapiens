# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.structures import PixelData
from mmengine.optim import OptimWrapper
from mmengine.model import BaseModule, ModuleList
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, ForwardResults, add_prefix)
from ..utils import resize
from .encoder_decoder import EncoderDecoder
from .base import BaseSegmentor
from ..builder import build_loss
from torch.nn import MultiheadAttention, LayerNorm, GELU
from mmpretrain.models.utils import CrossMultiheadAttention, build_norm_layer
from mmpretrain.models.backbones.vision_transformer import TransformerEncoderLayer
from .stereo_pointmap_estimator import StereoEncoderLayer

@MODELS.register_module()
class StereoCorrespondencesEstimator(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 embed_dims: int,
                 num_layers: int,
                 num_heads: int,
                 decode_head1: ConfigType,
                 decode_head2: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 loss_stereo_decode=dict(
                     type='StereoPointmapCorrespondenceLoss',
                     loss_weight=1.0),
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):

        BaseSegmentor.__init__(self, data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_layers = num_layers

        num_patches = self.backbone.patch_resolution[0] * self.backbone.patch_resolution[1]

        self.pos_embed1 = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))

        self.layers = ModuleList()

        for i in range(self.num_layers):
            self.layers.append(StereoEncoderLayer(self.embed_dims, self.num_heads, self.backbone.patch_resolution))

        assert neck is None
        assert auxiliary_head is None

        self._init_decode_head(decode_head1, decode_head2)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if isinstance(loss_stereo_decode, dict):
            self.loss_stereo_decode = build_loss(loss_stereo_decode)
        elif isinstance(loss_stereo_decode, (list, tuple)):
            self.loss_stereo_decode = nn.ModuleList()
            for loss in loss_stereo_decode:
                self.loss_stereo_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_stereo_decode must be a dict or sequence of dict,\
                but got {type(loss_stereo_decode)}')

        return

    def _init_decode_head(self, decode_head1: ConfigType, decode_head2: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head1 = MODELS.build(decode_head1)
        self.decode_head2 = MODELS.build(decode_head2)

        self.align_corners = self.decode_head1.align_corners
        self.num_classes = self.decode_head1.num_classes
        self.out_channels = self.decode_head1.out_channels

    def forward(self,
                inputs1: Tensor,
                inputs2: Tensor,
                data_samples1: OptSampleList = None,
                data_samples2: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode == 'loss':
            return self.loss(inputs1, inputs2, data_samples1, data_samples2)
        elif mode == 'predict':
            return self.predict(inputs1, inputs2, data_samples1, data_samples2)
        elif mode == 'tensor':
            return self._forward(inputs1, inputs2, data_samples1, data_samples2)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_feat(self,
                     inputs1: Tensor,
                     inputs2: Tensor,
                     batch_img_metas1: Optional[List[dict]] = None,
                     batch_img_metas2: Optional[List[dict]] = None) -> Tensor:
        x1 = self.backbone(inputs1)[0] # [B, C, H, W]
        x2 = self.backbone(inputs2)[0] # [B, C, H, W]

        B, C, H, W = x1.shape

        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        x1 = x1.reshape(B, C, -1).permute(0, 2, 1) # (B, N, C)
        x2 = x2.reshape(B, C, -1).permute(0, 2, 1) # (B, N, C)

        x1 = x1 + self.pos_embed1
        x2 = x2 + self.pos_embed2

        # # Apply attention blocks
        for block in self.layers:
            x1, x2 = block(x1, x2)

        ## reshape to featmap
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        x1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W)

        return tuple([x1]), tuple([x2])

    def encode_decode(self, inputs1: Tensor, inputs2: Tensor,
                      batch_img_metas1: List[dict], batch_img_metas2: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a depth map of the same
        size as input."""
        x1, x2 = self.extract_feat(inputs1, inputs2, batch_img_metas1, batch_img_metas2)

        desc1 = self.decode_head1.predict(x1, batch_img_metas1, self.test_cfg)
        desc2 = self.decode_head2.predict(x2, batch_img_metas2, self.test_cfg)
        return desc1, desc2

    def _decode_head_forward_train(self, inputs1: List[Tensor],
                                   inputs2: List[Tensor],
                                   data_samples1: SampleList,
                                   data_samples2: SampleList,) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode1, feats1 = self.decode_head1.loss(inputs1, data_samples1, self.train_cfg)
        loss_decode2, feats2 = self.decode_head2.loss(inputs2, data_samples2, self.train_cfg)

        losses.update(add_prefix(loss_decode1, 'decode1'))
        losses.update(add_prefix(loss_decode2, 'decode2'))

        ## compute correspondences loss
        correspondences_loss = self.loss_by_feat(feats1, feats2, data_samples1, data_samples2)
        losses.update(add_prefix(correspondences_loss, 'stereo'))

        return losses, feats1, feats2

    def loss_by_feat(self, feats1: Tensor, feats2: Tensor,
                     batch_data_samples1: SampleList,
                     batch_data_samples2: SampleList) -> dict:

        device = feats1.device
        B, C, H, W = feats1.shape

        gt_size = batch_data_samples1[0].img_shape ## 1024 x 768

        ## upsample
        feats1 = F.interpolate(feats1, size=gt_size, mode='bilinear', align_corners=self.decode_head1.align_corners) # B x 32 x 1024 x 768. 32 is dim
        feats2 = F.interpolate(feats2, size=gt_size, mode='bilinear', align_corners=self.decode_head1.align_corners) # B x 32 x 1024 x 768. 32 is dim

        loss = dict()

        if not isinstance(self.loss_stereo_decode, nn.ModuleList):
            losses_stereo_decode = [self.loss_stereo_decode]
        else:
            losses_stereo_decode = self.loss_stereo_decode

        for loss_stereo_decode in losses_stereo_decode:
            this_loss = loss_stereo_decode(
                        feats1,
                        feats2,
                        batch_data_samples1,
                        weight=None,)

            if loss_stereo_decode.loss_name not in loss:
                loss[loss_stereo_decode.loss_name] = this_loss
            else:
                loss[loss_stereo_decode.loss_name] += this_loss

        return loss

    def loss(self, inputs1: Tensor, inputs2: Tensor, data_samples1: SampleList, data_samples2: SampleList) -> dict:
        assert data_samples1 is not None
        assert data_samples2 is not None

        batch_img_metas1 = [
            data_sample.metainfo for data_sample in data_samples1
        ]
        batch_img_metas2 = [
            data_sample.metainfo for data_sample in data_samples2
        ]

        x1, x2 = self.extract_feat(inputs1, inputs2, batch_img_metas1, batch_img_metas2)

        losses = dict()

        loss_decode, seg_logits1, seg_logits2 = self._decode_head_forward_train(x1, x2, data_samples1, data_samples2)
        losses.update(loss_decode)

        return losses, seg_logits1, seg_logits2

    def predict(self,
                inputs1: Tensor,
                inputs2: Tensor,
                data_samples1: OptSampleList = None,
                data_samples2: OptSampleList = None) -> SampleList:
        if data_samples1 is not None:
            batch_img_metas1 = [
                data_sample.metainfo for data_sample in data_samples1
            ]
        else:
            batch_img_metas1 = [
                dict(
                    ori_shape=inputs1.shape[2:],
                    img_shape=inputs1.shape[2:],
                    pad_shape=inputs1.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs1.shape[0]

        if data_samples2 is not None:
            batch_img_metas2 = [
                data_sample.metainfo for data_sample in data_samples2
            ]
        else:
            batch_img_metas2 = [
                dict(
                    ori_shape=inputs2.shape[2:],
                    img_shape=inputs2.shape[2:],
                    pad_shape=inputs2.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs2.shape[0]

        desc1, desc2 = self.inference(inputs1, inputs2, batch_img_metas1, batch_img_metas2)

        return self.postprocess_result(desc1, desc2, data_samples1, data_samples2)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_depth_map`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def inference(self, inputs1: Tensor, inputs2: Tensor, batch_img_metas1: List[dict], batch_img_metas2: List[dict]) -> Tensor:
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole',
                                                      'slide_flip'], \
            f'Only "slide", "slide_flip" or "whole" test mode are ' \
            f'supported, but got {self.test_cfg["mode"]}.'

        ori_shape1 = batch_img_metas1[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape1 for _ in batch_img_metas1):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)

        ori_shape2 = batch_img_metas2[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape2 for _ in batch_img_metas2):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)

        desc1, desc2 = self.encode_decode(inputs1, inputs2, batch_img_metas1, batch_img_metas2)
        return desc1, desc2

    def postprocess_result(self,
                           desc1: Tensor,
                           desc2: Tensor,
                           data_samples1: OptSampleList = None,
                           data_samples2: OptSampleList = None) -> SampleList:

        batch_size, C, H, W = desc1.shape

        assert desc1.shape == desc2.shape

        if data_samples1 is None:
            data_samples1 = [SegDataSample() for _ in range(batch_size)]
            data_samples2 = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        data_samples = data_samples1.copy()

        for i in range(batch_size):
            if not only_prediction:
                img_meta1 = data_samples1[i].metainfo
                img_meta2 = data_samples2[i].metainfo

                # remove padding area
                if 'img_padding_size' not in img_meta1:
                    padding_size1 = img_meta1.get('padding_size', [0] * 4)
                else:
                    padding_size1 = img_meta1['img_padding_size']

                if 'img_padding_size' not in img_meta2:
                    padding_size2 = img_meta2.get('padding_size', [0] * 4)
                else:
                    padding_size2 = img_meta2['img_padding_size']

                assert padding_size1 == padding_size2
                assert padding_size1[0] == 0 and padding_size1[1] == 0 and padding_size1[2] == 0 and padding_size1[3] == 0

                i_desc1 = desc1[i:i + 1, :, :, :]
                i_desc2 = desc2[i:i + 1, :, :, :]

                ## resize to original image size. turned off for now
                i_desc1 = F.interpolate(i_desc1, size=img_meta1['ori_shape'], mode='bilinear', align_corners=self.align_corners) 
                i_desc2 = F.interpolate(i_desc2, size=img_meta2['ori_shape'], mode='bilinear', align_corners=self.align_corners)

                i_desc1 = i_desc1.squeeze(0) ## 32 X H x W
                i_desc2 = i_desc2.squeeze(0) ## 32 X H x W

            else:
                i_desc1 = desc1[i]
                i_desc2 = desc2[i]

            data_samples[i].set_data({
                'pred_desc1': PixelData(**{'data': i_desc1}),
                'pred_desc2': PixelData(**{'data': i_desc2}),
            })

        return data_samples


    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses, desc1, desc2 = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)

        log_vars['vis_preds'] = (desc1, desc2)
        return log_vars


    def postprocess_train_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo

                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size

                assert padding_left == 0 and \
                    padding_right == 0 and\
                    padding_top == 0 and\
                    padding_bottom == 0

                i_seg_logits = seg_logits[i:i + 1, :, :]
                i_seg_logits = F.interpolate(i_seg_logits, size=img_meta['img_shape'], mode='bilinear', align_corners=self.align_corners)
                i_seg_logits = i_seg_logits.squeeze(0) ## 32 X H x W

            else:
                i_seg_logits = seg_logits[i]

            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
            })

        return data_samples
