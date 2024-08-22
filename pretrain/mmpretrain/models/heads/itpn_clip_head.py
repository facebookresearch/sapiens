from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.device import get_device
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class iTPNClipHead(BaseModule):
    """Head for iTPN Pre-training using Clip.

    Compute the logits and the cross entropy loss.

    Args:
        embed_dims (int): The dimension of embedding.
        num_embed (int): The number of classification types.
        loss (dict): The config of loss.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_embed: int,
        loss: dict,
        init_cfg: Optional[Union[dict, List[dict]]] = dict(
            type='TruncNormal', layer='Linear', std=0.02, bias=0)
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.cls_head = nn.Linear(embed_dims, num_embed)
        self.loss_module = MODELS.build(loss)

    def loss(self, feats: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            feats (torch.Tensor): Features from backbone.
            target (torch.Tensor): Target generated by target_generator.
            mask (torch.Tensor): Generated mask for pretraing.
        """
        mask = mask.to(get_device(), non_blocking=True)
        mask = mask.flatten(1).to(torch.bool)
        target = target[mask]

        # remove cls_token
        # feats = feats[:, 1:]
        logits = self.cls_head(feats[mask])

        loss = self.loss_module(logits, target)
        return loss