# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.models import SwinTransformer
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor


@MODELS.register_module()
class SimMIMSwinTransformer(SwinTransformer):
    """Swin Transformer for SimMIM pre-training.

    Args:
        Args:
        arch (str | dict): Swin Transformer architecture
            Defaults to 'T'.
        img_size (int | tuple): The size of input image.
            Defaults to 224.
        in_channels (int): The num of input channels.
            Defaults to 3.
        drop_rate (float): Dropout rate after embedding.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate.
            Defaults to 0.1.
        out_indices (tuple): Layers to be outputted. Defaults to (3, ).
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer at end
            of backbone. Defaults to dict(type='LN')
        stage_cfgs (Sequence | dict): Extra config dict for each
            stage. Defaults to empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to empty dict.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 arch: Union[str, dict] = 'T',
                 img_size: Union[Tuple[int, int], int] = 224,
                 in_channels: int = 3,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 out_indices: tuple = (3, ),
                 use_abs_pos_embed: bool = False,
                 with_cp: bool = False,
                 frozen_stages: bool = -1,
                 norm_eval: bool = False,
                 norm_cfg: dict = dict(type='LN'),
                 stage_cfgs: Union[Sequence, dict] = dict(),
                 patch_cfg: dict = dict(),
                 pad_small_map: bool = False,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            in_channels=in_channels,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            use_abs_pos_embed=use_abs_pos_embed,
            with_cp=with_cp,
            frozen_stages=frozen_stages,
            norm_eval=norm_eval,
            norm_cfg=norm_cfg,
            stage_cfgs=stage_cfgs,
            patch_cfg=patch_cfg,
            pad_small_map=pad_small_map,
            init_cfg=init_cfg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        if self.use_abs_pos_embed:
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        trunc_normal_(self.mask_token, mean=0, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor, optional): Masks for images.

        Returns:
            tuple: A tuple containing features from multi-stages.
        """
        if mask is None:
            return super().forward(x)

        else:
            x, hw_shape = self.patch_embed(x)
            B, L, _ = x.shape

            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            if self.use_abs_pos_embed:
                x = x + self.absolute_pos_embed

            x = self.drop_after_pos(x)

            outs = []
            for i, stage in enumerate(self.stages):
                x, hw_shape = stage(x, hw_shape)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(x)
                    out = out.view(-1, *hw_shape,
                                   stage.out_channels).permute(0, 3, 1,
                                                               2).contiguous()
                    outs.append(out)

            return tuple(outs)


@MODELS.register_module()
class SimMIM(BaseSelfSupervisor):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
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

        img_latent = self.backbone(inputs, mask)
        img_rec = self.neck(img_latent[0])
        loss = self.head.loss(img_rec, inputs, mask)
        losses = dict(loss=loss)

        return losses
