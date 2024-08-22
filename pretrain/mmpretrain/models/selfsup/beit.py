# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from torch import nn

from mmpretrain.models.backbones import BEiTViT
from mmpretrain.models.utils import NormEMAVectorQuantizer, resize_pos_embed
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor


@MODELS.register_module()
class VQKD(BaseModule):
    """Vector-Quantized Knowledge Distillation.

    The module only contains encoder and VectorQuantizer part
    Modified from https://github.com/microsoft/unilm/blob/master/beit2/modeling_vqkd.py

    Args:
        encoder_config (dict): The config of encoder.
        decoder_config (dict, optional): The config of decoder. Currently,
            VQKD only support to build encoder. Defaults to None.
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        beta (float): The mutiplier for VectorQuantizer loss. Defaults to 1.
        quantize_kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 encoder_config: dict,
                 decoder_config: Optional[dict] = None,
                 num_embed: int = 8192,
                 embed_dims: int = 32,
                 decay: float = 0.99,
                 beta: float = 1.0,
                 quantize_kmeans_init: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.encoder = BEiTViT(**encoder_config)
        if decoder_config is not None:
            self.decoder = BEiTViT(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            num_embed=num_embed,
            embed_dims=embed_dims,
            beta=beta,
            decay=decay,
            kmeans_init=quantize_kmeans_init,
        )

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder.arch_settings['embed_dims'],
                      self.encoder.arch_settings['embed_dims']), nn.Tanh(),
            nn.Linear(self.encoder.arch_settings['embed_dims'], embed_dims))

    def get_tokens(self, x: torch.Tensor) -> dict:
        """Get tokens for beit pre-training."""
        _, embed_ind, _ = self.encode(x)
        output = {}
        output['token'] = embed_ind.view(x.shape[0], -1)
        output['input_img'] = x

        return output

    def encode(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input images and get corresponding results."""
        encoder_features = self.encoder(x)[0]
        B, C, N1, N2 = encoder_features.shape
        encoder_features = encoder_features.permute(0, 2, 3,
                                                    1).reshape(B, N1 * N2, C)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(
                encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = int(math.sqrt(N)), int(math.sqrt(N))

        to_quantizer_features = rearrange(
            to_quantizer_features, 'b (h w) c -> b c h w', h=h,
            w=w)  # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function.

        Currently, only support to get tokens.
        """
        return self.get_tokens(x)['token']


@MODELS.register_module()
class BEiTPretrainViT(BEiTViT):
    """Vision Transformer for BEiT pre-training.

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            It only works without input mask. Defaults to ``"avg_featmap"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        use_abs_pos_emb (bool): Whether or not use absolute position embedding.
            Defaults to False.
        use_rel_pos_bias (bool): Whether or not use relative position bias.
            Defaults to False.
        use_shared_rel_pos_bias (bool): Whether or not use shared relative
            position bias. Defaults to True.
        layer_scale_init_value (float): The initialization value for
            the learnable scaling of attention and FFN. Defaults to 0.1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_indices: int = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 out_type: str = 'raw',
                 frozen_stages: int = -1,
                 use_abs_pos_emb: bool = False,
                 use_rel_pos_bias: bool = False,
                 use_shared_rel_pos_bias: bool = True,
                 layer_scale_init_value: int = 0.1,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(padding=0),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            out_type=out_type,
            with_cls_token=True,
            frozen_stages=frozen_stages,
            use_abs_pos_emb=use_abs_pos_emb,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            use_rel_pos_bias=use_rel_pos_bias,
            layer_scale_init_value=layer_scale_init_value,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        self.rescale_init_weight()

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.ffn.layers[1].weight.data, layer_id + 1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor]:
        """The BEiT style forward function.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images, which is of shape (B x C x H x W).
            mask (torch.Tensor, optional): Mask for input, which is of shape
                (B x patch_resolution[0] x patch_resolution[1]).

        Returns:
            Tuple[torch.Tensor]: Hidden features.
        """
        if mask is None:
            return super().forward(x)

        else:
            x, patch_resolution = self.patch_embed(x)

            # replace the masked visual tokens by mask_token
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                x = x + resize_pos_embed(
                    self.pos_embed,
                    self.patch_resolution,
                    patch_resolution,
                    mode=self.interpolate_mode,
                    num_extra_tokens=self.num_extra_tokens)
            x = self.drop_after_pos(x)

            self.shared_rel_pos_bias = self.rel_pos_bias().to(
                mask.device) if self.rel_pos_bias is not None else None

            outs = []
            for i, layer in enumerate(self.layers):
                x = layer(x, rel_pos_bias=self.shared_rel_pos_bias)

                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.norm1(x)

                if i in self.out_indices:
                    outs.append(x)

            return tuple(outs)


@MODELS.register_module()
class BEiT(BaseSelfSupervisor):
    """BEiT v1/v2.

    Implementation of `BEiT: BERT Pre-Training of Image Transformers
    <https://arxiv.org/abs/2106.08254>`_ and `BEiT v2: Masked Image Modeling
    with Vector-Quantized Visual Tokenizers
    <https://arxiv.org/abs/2208.06366>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        mask = torch.stack([data_sample.mask for data_sample in data_samples])

        img_latent = self.backbone(inputs[0], mask)

        # inputs[1] is the target image
        with torch.no_grad():
            target = self.target_generator(inputs[1])
            target = target.detach()

        if self.with_neck:
            # BEiT v2
            feats, feats_cls_pt = self.neck(
                img_latent, rel_pos_bias=self.backbone.shared_rel_pos_bias)
            loss = self.head.loss(feats, feats_cls_pt, target, mask)
        else:
            # BEiT v1
            loss = self.head.loss(img_latent[0], target, mask)

        if isinstance(loss, torch.Tensor):
            losses = dict(loss=loss)
            return losses
        elif isinstance(loss, Tuple):
            # the loss_1 and loss_2 are general reconstruction loss (patch
            # feature vectors from last layer of backbone) and early state
            # reconstruction loss (patch feature vectors from intermediate
            # layer of backbone)
            loss_1, loss_2 = loss[0], loss[1]
            losses = dict()
            # the key with prefix 'loss', like loss_1 and loss_2, will be used
            # as the final criterion
            losses['loss_1'] = loss_1
            losses['loss_2'] = loss_2
            return losses
