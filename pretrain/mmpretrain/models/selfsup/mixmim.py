# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmpretrain.models.backbones import MixMIMTransformer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor


@MODELS.register_module()
class MixMIMPretrainTransformer(MixMIMTransformer):
    """MixMIM backbone for MixMIM pre-training.

    A PyTorch implement of : ` MixMIM: Mixed and Masked Image
    Modeling for Efficient Visual Representation Learning
    <https://arxiv.org/abs/2205.13137>`_

    Args:
        arch (str | dict): MixMIM architecture. If use string,
            choose from 'base','large' and 'huge'.
            If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.

            Defaults to 'base'.
        mlp_ratio (int): The mlp ratio in FFN.  Defaults to 4.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to mlp_ratio
            the most common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (list): The height and width of the window.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        attn_drop_rate (float): Attention drop rate. Defaults to 0.
        use_checkpoint (bool): Whether use the checkpoint to reduce GPU memory
            cost. Defaults to False.
        mask_ratio (bool): The base ratio of total number of patches to be
            masked. Defaults to 0.5.
        range_mask_ratio (float): The range of mask ratio.
            Defaults to 0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'base',
                 mlp_ratio: float = 4,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 window_size: List = [14, 14, 14, 7],
                 qkv_bias: bool = True,
                 patch_cfg: dict = dict(),
                 norm_cfg: dict = dict(type='LN'),
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 use_checkpoint: bool = False,
                 mask_ratio: float = 0.5,
                 range_mask_ratio: float = 0.0,
                 init_cfg: Optional[dict] = None) -> None:

        super().__init__(
            arch=arch,
            mlp_ratio=mlp_ratio,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            window_size=window_size,
            qkv_bias=qkv_bias,
            patch_cfg=patch_cfg,
            norm_cfg=norm_cfg,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            init_cfg=init_cfg)

        self.mask_ratio = mask_ratio
        self.range_mask_ratio = range_mask_ratio

    def init_weights(self):
        """Initialize position embedding, patch embedding."""
        super(MixMIMTransformer, self).init_weights()

        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.absolute_pos_embed.shape[-1],
            cls_token=False)
        self.absolute_pos_embed.data.copy_(pos_embed.float())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self,
                       x: torch.Tensor,
                       mask_ratio: float = 0.5) -> Tuple[torch.Tensor]:
        """Generate the mask for MixMIM Pretraining.

        Args:
            x (torch.Tensor): Image with data augmentation applied, which is
                of shape B x L x C.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.5.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - mask_s1 (torch.Tensor): mask with stride of
                  self.encoder_stride // 8.
                - mask_s2 (torch.Tensor): mask with stride of
                  self.encoder_stride // 4.
                - mask_s3 (torch.Tensor): mask with stride of
                  self.encoder_stride // 2.
                - mask (torch.Tensor): mask with stride of
                  self.encoder_stride.
        """

        B, C, H, W = x.shape
        out_H = H // self.encoder_stride
        out_W = W // self.encoder_stride
        s3_H, s3_W = out_H * 2, out_W * 2
        s2_H, s2_W = out_H * 4, out_W * 4
        s1_H, s1_W = out_H * 8, out_W * 8

        seq_l = out_H * out_W
        # use a shared mask for a batch images
        mask = torch.zeros([1, 1, seq_l], device=x.device)

        mask_ratio = mask_ratio + random.uniform(0.0, self.range_mask_ratio)
        noise = torch.rand(1, 1, seq_l, device=x.device)  # noise in [0, 1]
        # ascend: small is keep, large is removed
        mask_idx = torch.argsort(noise, dim=2)[:, :, :int(seq_l * mask_ratio)]
        mask.scatter_(2, mask_idx, 1)
        mask = mask.reshape(1, 1, out_H, out_W)
        mask_s1 = F.interpolate(mask, size=(s1_H, s1_W), mode='nearest')
        mask_s2 = F.interpolate(mask, size=(s2_H, s2_W), mode='nearest')
        mask_s3 = F.interpolate(mask, size=(s3_H, s3_W), mode='nearest')

        mask = mask.reshape(1, out_H * out_W, 1).contiguous()
        mask_s1 = mask_s1.reshape(1, s1_H * s1_W, 1).contiguous()
        mask_s2 = mask_s2.reshape(1, s2_H * s2_W, 1).contiguous()
        mask_s3 = mask_s3.reshape(1, s3_H * s3_W, 1).contiguous()

        return mask_s1, mask_s2, mask_s3, mask

    def forward(self,
                x: torch.Tensor,
                mask: Optional[bool] = True) -> Tuple[torch.Tensor]:
        """Generate features for masked images.

        This function generates mask and masks some patches randomly and get
        the hidden features for visible patches.

        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward containing
                ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - x (torch.Tensor): hidden features, which is of shape
                B x L x C.
              - mask_s4 (torch.Tensor): the mask tensor for the last layer.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            mask_s1, mask_s2, mask_s3, mask_s4 = self.random_masking(
                x, self.mask_ratio)

            x, _ = self.patch_embed(x)

            x = x * (1. - mask_s1) + x.flip(0) * mask_s1
            x = x + self.absolute_pos_embed
            x = self.drop_after_pos(x)

            for idx, layer in enumerate(self.layers):
                if idx == 0:
                    x = layer(x, attn_mask=mask_s1)
                elif idx == 1:
                    x = layer(x, attn_mask=mask_s2)
                elif idx == 2:
                    x = layer(x, attn_mask=mask_s3)
                elif idx == 3:
                    x = layer(x, attn_mask=mask_s4)

            x = self.norm(x)

            return x, mask_s4


@MODELS.register_module()
class MixMIM(BaseSelfSupervisor):
    """MixMIM.

    Implementation of `MixMIM: Mixed and Masked Image Modeling for Efficient
    Visual Representation Learning. <https://arxiv.org/abs/2205.13137>`_.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):

        head.update(dict(patch_size=neck['encoder_stride']))
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        latent, mask = self.backbone(inputs)
        x_rec = self.neck(latent, mask)
        loss = self.head.loss(x_rec, inputs, mask)
        losses = dict(loss=loss)
        return losses
