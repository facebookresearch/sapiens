# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import torch
from mmengine.dist import all_gather, get_rank

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseSelfSupervisor


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out


@MODELS.register_module()
class SimCLR(BaseSelfSupervisor):
    """SimCLR.

    Implementation of `A Simple Framework for Contrastive Learning of Visual
    Representations <https://arxiv.org/abs/2002.05709>`_.
    """

    @staticmethod
    def _create_buffer(
        batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the mask and the index of positive samples.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device of backend.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The mask for feature selection.
            - The index of positive samples.
            - The mask of negative samples.
        """
        mask = 1 - torch.eye(batch_size * 2, dtype=torch.uint8).to(device)
        pos_idx = (
            torch.arange(batch_size * 2).to(device),
            2 * torch.arange(batch_size, dtype=torch.long).unsqueeze(1).repeat(
                1, 2).view(-1, 1).squeeze().to(device))
        neg_mask = torch.ones((batch_size * 2, batch_size * 2 - 1),
                              dtype=torch.uint8).to(device)
        neg_mask[pos_idx] = 0
        return mask, pos_idx, neg_mask

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
        inputs = torch.stack(inputs, 1)
        inputs = inputs.reshape((inputs.size(0) * 2, inputs.size(2),
                                 inputs.size(3), inputs.size(4)))
        x = self.backbone(inputs)
        z = self.neck(x)[0]  # (2n)xd

        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_idx, neg_mask = self._create_buffer(N, s.device)

        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_idx].unsqueeze(1)  # (2N)x1

        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)

        loss = self.head.loss(positive, negative)
        losses = dict(loss=loss)
        return losses
