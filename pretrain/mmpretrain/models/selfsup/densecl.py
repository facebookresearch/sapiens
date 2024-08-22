# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.dist import all_gather
from mmengine.model import ExponentialMovingAverage

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from ..utils import batch_shuffle_ddp, batch_unshuffle_ddp
from .base import BaseSelfSupervisor


@MODELS.register_module()
class DenseCL(BaseSelfSupervisor):
    """DenseCL.

    Implementation of `Dense Contrastive Learning for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2011.09157>`_.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL>`_.
    The loss_lambda warmup is in `engine/hooks/densecl_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors.
        head (dict): Config dict for module of head functions.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
        loss_lambda (float): Loss weight for the single and dense contrastive
            loss. Defaults to 0.5.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 queue_len: int = 65536,
                 feat_dim: int = 128,
                 momentum: float = 0.001,
                 loss_lambda: float = 0.5,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.encoder_k = ExponentialMovingAverage(
            nn.Sequential(self.backbone, self.neck), momentum)

        self.queue_len = queue_len
        self.loss_lambda = loss_lambda

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # create the second queue for dense output
        self.register_buffer('queue2', torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer('queue2_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update queue."""
        # gather keys before updating queue
        keys = torch.cat(all_gather(keys), dim=0)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys: torch.Tensor) -> None:
        """Update queue2."""
        # gather keys before updating queue
        keys = torch.cat(all_gather(keys), dim=0)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

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
        assert isinstance(inputs, list)
        im_q = inputs[0]
        im_k = inputs[1]
        # compute query features
        q_b = self.backbone(im_q)  # backbone features
        q, q_grid, q2 = self.neck(q_b)  # queries: NxC; NxCxS^2
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self.encoder_k.update_parameters(
                nn.Sequential(self.backbone, self.neck))

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k_b = self.encoder_k.module[0](im_k)  # backbone features
            k, k_grid, k2 = self.encoder_k.module[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2,
                                      densecl_sim_ind.unsqueeze(1).expand(
                                          -1, k_grid.size(1), -1))  # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        # dense positive logits: NS^2X1
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        # dense negative logits: NS^2xK
        l_neg_dense = torch.einsum(
            'nc,ck->nk', [q_grid, self.queue2.clone().detach()])

        loss_single = self.head.loss(l_pos, l_neg)
        loss_dense = self.head.loss(l_pos_dense, l_neg_dense)

        losses = dict()
        losses['loss_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_dense'] = loss_dense * self.loss_lambda

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        return losses
