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

class StereoEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 patch_resolution,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None
                 ):
        super(StereoEncoderLayer, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.patch_resolution = patch_resolution

        num_patches = patch_resolution[0] * patch_resolution[1]
        feedforward_channels = 4 * embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        ##---------------------------------------------------
        self.cross_attention1 = CrossMultiheadAttention(embed_dims, num_heads, qkv_bias=True)
        self.cross_attention2 = CrossMultiheadAttention(embed_dims, num_heads, qkv_bias=True)

        self.self_attention1 = TransformerEncoderLayer(
                    embed_dims,
                    num_heads,
                    feedforward_channels,
                    layer_scale_init_value=0.,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.,
                    num_fcs=2,
                    qkv_bias=True,
                    ffn_type='origin',
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN', eps=1e-6),
                    init_cfg=None)

        self.self_attention2 = TransformerEncoderLayer(
                    embed_dims,
                    num_heads,
                    feedforward_channels,
                    layer_scale_init_value=0.,
                    drop_rate=0.,
                    attn_drop_rate=0.,
                    drop_path_rate=0.,
                    num_fcs=2,
                    qkv_bias=True,
                    ffn_type='origin',
                    act_cfg=dict(type='GELU'),
                    norm_cfg=dict(type='LN', eps=1e-6),
                    init_cfg=None)
        return

    def init_weights(self):
        self.self_attention1.init_weights()
        self.self_attention2.init_weights()
        self.cross_attention1.init_weights()
        self.cross_attention2.init_weights()
        return

    # https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/blocks.py#L186
    def forward(self, x1, x2):
        x1_ = self.ln1(x1)
        x2_ = self.ln2(x2)

        x1 = x1 + self.cross_attention1(x1, x2_, x2_) ## q, k, v
        x2 = x2 + self.cross_attention2(x2, x1_, x1_) ## q, k, v

        x1 = self.self_attention1(x1) ## is residual already
        x2 = self.self_attention2(x2)

        return x1, x2


@MODELS.register_module()
class StereoPointmapEstimator(EncoderDecoder):
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
        depth1 = self.decode_head1.predict(x1, batch_img_metas1, self.test_cfg)
        depth2 = self.decode_head2.predict(x2, batch_img_metas2, self.test_cfg)
        return depth1, depth2

    def _decode_head_forward_train(self, inputs1: List[Tensor],
                                   inputs2: List[Tensor],
                                   data_samples1: SampleList,
                                   data_samples2: SampleList,) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode1, seg_logits1 = self.decode_head1.loss(inputs1, data_samples1, self.train_cfg)
        loss_decode2, seg_logits2 = self.decode_head2.loss(inputs2, data_samples2, self.train_cfg)

        losses.update(add_prefix(loss_decode1, 'decode1'))
        losses.update(add_prefix(loss_decode2, 'decode2'))

        ## compute correspondence loss
        stereo_loss = self.loss_by_feat(seg_logits1, seg_logits2, data_samples1, data_samples2)
        losses.update(add_prefix(stereo_loss, 'stereo'))

        return losses, seg_logits1, seg_logits2

    def loss_by_feat(self, seg_logits1: Tensor, seg_logits2: Tensor,
                     batch_data_samples1: SampleList,
                     batch_data_samples2: SampleList) -> dict:

        device = seg_logits1.device
        B, C, H, W = seg_logits1.shape

        gt_size = batch_data_samples1[0].gt_depth_map.shape

        ## upsample
        seg_logits1 = F.interpolate(seg_logits1, size=gt_size, mode='bilinear', align_corners=self.decode_head1.align_corners)
        seg_logits2 = F.interpolate(seg_logits2, size=gt_size, mode='bilinear', align_corners=self.decode_head1.align_corners)

        # ## debug
        # seg_logits1 = [data_sample.gt_depth_map.data.to(device) for data_sample in batch_data_samples1]
        # seg_logits1 = torch.stack(seg_logits1, dim=0) ## B x 3 x H x W
        # seg_logits2 = [data_sample.gt_depth_map.data.to(device) for data_sample in batch_data_samples2]
        # seg_logits2 = torch.stack(seg_logits2, dim=0) ## B x 3 x H x W

        loss = dict()

        if not isinstance(self.loss_stereo_decode, nn.ModuleList):
            losses_stereo_decode = [self.loss_stereo_decode]
        else:
            losses_stereo_decode = self.loss_stereo_decode

        for loss_stereo_decode in losses_stereo_decode:
            this_loss = loss_stereo_decode(
                        seg_logits1,
                        seg_logits2,
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

        depth1, depth2 = self.inference(inputs1, inputs2, batch_img_metas1, batch_img_metas2)

        return self.postprocess_result(depth1, depth2, data_samples1, data_samples2)

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

        depth1, depth2 = self.encode_decode(inputs1, inputs2, batch_img_metas1, batch_img_metas2)
        return depth1, depth2

    def postprocess_result(self,
                           depth1: Tensor,
                           depth2: Tensor,
                           data_samples1: OptSampleList = None,
                           data_samples2: OptSampleList = None) -> SampleList:

        batch_size, C, H, W = depth1.shape

        assert depth1.shape == depth2.shape

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

                padding_left1, padding_right1, padding_top1, padding_bottom1 = padding_size1
                padding_left2, padding_right2, padding_top2, padding_bottom2 = padding_size2

                # i_depth shape is 1, C, H, W after remove padding
                i_depth1 = depth1[i:i + 1, :, padding_top1:H - padding_bottom1, padding_left1:W - padding_right1]
                i_depth2 = depth2[i:i + 1, :, padding_top2:H - padding_bottom2, padding_left2:W - padding_right2]

                ## padding is actually all zero. i_depth is B x 3 x 1024 x 768
                flip1 = img_meta1.get('flip', None)
                flip2 = img_meta2.get('flip', None)

                if flip1:
                    flip_direction1 = img_meta1.get('flip_direction', None)
                    assert flip_direction1 in ['horizontal', 'vertical']
                    if flip_direction1 == 'horizontal':
                        i_depth1 = i_depth1.flip(dims=(3, ))
                    else:
                        i_depth1 = i_depth1.flip(dims=(2, ))

                if flip2:
                    flip_direction2 = img_meta1.get('flip_direction', None)
                    assert flip_direction2 in ['horizontal', 'vertical']
                    if flip_direction2 == 'horizontal':
                        i_depth2 = i_depth2.flip(dims=(3, ))
                    else:
                        i_depth2 = i_depth2.flip(dims=(2, ))

                ## resize pointmap1 as original shape
                i_depth_X1 = i_depth1[:, 0].unsqueeze(1) ## B x 1 X H x W
                i_depth_Y1 = i_depth1[:, 1].unsqueeze(1) ## B x 1 X H x W
                i_depth_Z1 = i_depth1[:, 2].unsqueeze(1) ## B x 1 X H x W

                i_depth_resized_X1 = F.interpolate(i_depth_X1, size=img_meta1['ori_shape'], mode='bilinear', align_corners=self.align_corners)
                i_depth_resized_Y1 = F.interpolate(i_depth_Y1, size=img_meta1['ori_shape'], mode='bilinear', align_corners=self.align_corners)
                i_depth_resized_Z1 = F.interpolate(i_depth_Z1, size=img_meta1['ori_shape'], mode='bilinear', align_corners=self.align_corners)

                i_depth1 = torch.cat([i_depth_resized_X1, i_depth_resized_Y1, i_depth_resized_Z1], dim=1) ## B x 3 X H x W
                i_depth1 = i_depth1.squeeze(0) ## 3 X H x W

                ## resize pointmap2 as original shape
                i_depth_X2 = i_depth2[:, 0].unsqueeze(1)
                i_depth_Y2 = i_depth2[:, 1].unsqueeze(1)
                i_depth_Z2 = i_depth2[:, 2].unsqueeze(1)

                i_depth_resized_X2 = F.interpolate(i_depth_X2, size=img_meta2['ori_shape'], mode='bilinear', align_corners=self.align_corners)
                i_depth_resized_Y2 = F.interpolate(i_depth_Y2, size=img_meta2['ori_shape'], mode='bilinear', align_corners=self.align_corners)
                i_depth_resized_Z2 = F.interpolate(i_depth_Z2, size=img_meta2['ori_shape'], mode='bilinear', align_corners=self.align_corners)

                i_depth2 = torch.cat([i_depth_resized_X2, i_depth_resized_Y2, i_depth_resized_Z2], dim=1)
                i_depth2 = i_depth2.squeeze(0) ## 3 X H x W

            else:
                i_depth1 = depth1[i]
                i_depth2 = depth2[i]

            data_samples[i].set_data({
                'pred_depth_map1': PixelData(**{'data': i_depth1}),
                'pred_depth_map2': PixelData(**{'data': i_depth2}),
            })

        return data_samples


    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses, seg_logits1, seg_logits2 = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)

        log_vars['vis_preds'] = (seg_logits1, seg_logits2)
        return log_vars


    def postprocess_train_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
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
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                # resize as original shape
                i_seg_logits_X = i_seg_logits[:, 0].unsqueeze(1) ## B x 1 X H x W
                i_seg_logits_Y = i_seg_logits[:, 1].unsqueeze(1) ## B x 1 X H x W
                i_seg_logits_Z = i_seg_logits[:, 2].unsqueeze(1) ## B x 1 X H x W

                i_seg_logits_resized_X = F.interpolate(i_seg_logits_X, size=img_meta['img_shape'], mode='bilinear', align_corners=self.align_corners)
                i_seg_logits_resized_Y = F.interpolate(i_seg_logits_Y, size=img_meta['img_shape'], mode='bilinear', align_corners=self.align_corners)
                i_seg_logits_resized_Z = F.interpolate(i_seg_logits_Z, size=img_meta['img_shape'], mode='bilinear', align_corners=self.align_corners)

                i_seg_logits = torch.cat([i_seg_logits_resized_X, i_seg_logits_resized_Y, i_seg_logits_resized_Z], dim=1) ## B x 3 X H x W
                i_seg_logits = i_seg_logits.squeeze(0) ## 3 X H x W

            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples
