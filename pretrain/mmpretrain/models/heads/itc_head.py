# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import all_gather
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class ITCHead(BaseModule):
    """Image-text matching head for multi-modal pre-trained task. Adapted by
    BLIP, ALBEF. Normally used for retrieval task.

    Args:
        embed_dim (int): Embed channel size for queue.
        queue_size (int): Queue size for image and text. Defaults to 57600.
        temperature (float): Temperature to calculate the similarity.
            Defaults to 0.07.
        use_distill (bool): Whether to use distill to calculate loss.
            Defaults to True.
        alpha (float): Weight for momentum similarity. Defaults to 0.4.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dim: int,
                 queue_size: int = 57600,
                 temperature: float = 0.07,
                 use_distill: bool = True,
                 alpha: float = 0.4,
                 init_cfg: Optional[dict] = None):
        super(ITCHead, self).__init__(init_cfg=init_cfg)
        self.temp = nn.Parameter(temperature * torch.ones([]))
        self.use_distill = use_distill
        if self.use_distill:
            # create the queue
            self.register_buffer('image_queue',
                                 torch.randn(embed_dim, queue_size))
            self.register_buffer('text_queue',
                                 torch.randn(embed_dim, queue_size))
            self.register_buffer('idx_queue', torch.full((1, queue_size),
                                                         -100))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

            self.image_queue = F.normalize(self.image_queue, dim=0)
            self.text_queue = F.normalize(self.text_queue, dim=0)

            self.queue_size = queue_size
            # This value will be warmup by `WarmupParamHook`
            self.alpha = alpha

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        return feats[-1]

    def loss(self, feats: Tuple[torch.Tensor], data_samples, **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # The part can be traced by torch.fx
        img_feats, text_feats, img_feats_m, text_feats_m = self(feats)

        img_feats_all = torch.cat(
            [img_feats_m.t(),
             self.image_queue.clone().detach()], dim=1)
        text_feats_all = torch.cat(
            [text_feats_m.t(),
             self.text_queue.clone().detach()], dim=1)

        # The part can not be traced by torch.fx
        losses = self._get_loss(img_feats, text_feats, img_feats_m,
                                text_feats_m, img_feats_all, text_feats_all,
                                data_samples, **kwargs)
        return losses

    def _get_loss(self, img_feats, text_feats, img_feats_m, text_feats_m,
                  img_feats_all, text_feats_all, data_samples, **kwargs):
        """Unpack data samples and compute loss."""

        idx = torch.tensor([ds.image_id
                            for ds in data_samples]).to(img_feats.device)
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            if self.use_distill:
                sim_i2t_m = img_feats_m @ text_feats_all / self.temp
                sim_t2i_m = text_feats_m @ img_feats_all / self.temp

                sim_i2t_targets = (
                    self.alpha * F.softmax(sim_i2t_m, dim=1) +
                    (1 - self.alpha) * sim_targets)
                sim_t2i_targets = (
                    self.alpha * F.softmax(sim_t2i_m, dim=1) +
                    (1 - self.alpha) * sim_targets)

        sim_i2t = img_feats @ text_feats_all / self.temp
        sim_t2i = text_feats @ img_feats_all / self.temp

        if self.use_distill:
            loss_i2t = -torch.sum(
                F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(
                F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        else:
            loss_i2t = -torch.sum(
                F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(
                F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        # compute loss
        losses = dict()

        losses['itc_loss'] = (loss_i2t + loss_t2i) / 2
        self._dequeue_and_enqueue(img_feats_m, text_feats_m, idx)
        return losses

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs=None):
        # gather keys before updating queue
        image_feats = torch.cat(all_gather(image_feat))
        text_feats = torch.cat(all_gather(text_feat))

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T

        if idxs is not None:
            idxs = torch.cat(all_gather(idxs))
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr
