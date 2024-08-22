# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2022 Microsoft
# Modified from
# https://github.com/microsoft/unilm/blob/master/beit2/norm_ema_quantizer.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmengine.dist import all_reduce


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor,
                decay: torch.Tensor) -> None:
    """Update moving average."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def norm_ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor,
                     decay: torch.Tensor) -> None:
    """Update moving average with norm data."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
    moving_avg.data.copy_(F.normalize(moving_avg.data, p=2, dim=-1))


def sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Sample vectors according to the given number."""
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num, ), device=device)

    return samples[indices]


def kmeans(samples: torch.Tensor,
           num_clusters: int,
           num_iters: int = 10,
           use_cosine_sim: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run k-means algorithm."""
    dim, dtype, _ = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    """The codebook of embedding vectors.

    Args:
        num_tokens (int): Number of embedding vectors in the codebook.
        codebook_dim (int) : The dimension of embedding vectors in the
            codebook.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self,
                 num_tokens: int,
                 codebook_dim: int,
                 kmeans_init: bool = True,
                 codebook_init_path: Optional[str] = None):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        if codebook_init_path is None:
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = F.normalize(weight, p=2, dim=-1)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f'load init codebook weight from {codebook_init_path}')
            codebook_ckpt_weight = torch.load(
                codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data: torch.Tensor) -> None:
        """Initialize embedding vectors of codebook."""
        if self.initted:
            return
        print('Performing K-means init for codebook')
        embed, _ = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.initted.data.copy_(torch.Tensor([True]))

    def forward(self, embed_id: torch.Tensor) -> torch.Tensor:
        """Get embedding vectors."""
        return F.embedding(embed_id, self.weight)


class NormEMAVectorQuantizer(nn.Module):
    """Normed EMA vector quantizer module.

    Args:
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        beta (float): The mutiplier for VectorQuantizer embedding loss.
            Defaults to 1.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        statistic_code_usage (bool): Whether to use cluster_size to record
            statistic. Defaults to True.
        kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        codebook_init_path (str): The initialization checkpoint for codebook.
            Defaults to None.
    """

    def __init__(self,
                 num_embed: int,
                 embed_dims: int,
                 beta: float,
                 decay: float = 0.99,
                 statistic_code_usage: bool = True,
                 kmeans_init: bool = True,
                 codebook_init_path: Optional[str] = None) -> None:
        super().__init__()
        self.codebook_dim = embed_dims
        self.num_tokens = num_embed
        self.beta = beta
        self.decay = decay

        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = EmbeddingEMA(
            num_tokens=self.num_tokens,
            codebook_dim=self.codebook_dim,
            kmeans_init=kmeans_init,
            codebook_init_path=codebook_init_path)

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(num_embed))

    def reset_cluster_size(self, device):

        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

    def forward(self, z):
        """Forward function."""
        # reshape z -> (batch, height, width, channel)
        z = rearrange(z, 'b c h w -> b h w c')
        z = F.normalize(z, p=2, dim=-1)
        z_flattened = z.reshape(-1, self.codebook_dim)

        self.embedding.init_embed_(z_flattened)

        # 'n d -> d n'
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)

        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                all_reduce(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)

        if self.training and self.embedding.update:
            # update cluster size with EMA
            bins = encodings.sum(0)
            all_reduce(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            all_reduce(embed_sum)

            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = F.normalize(embed_normalized, p=2, dim=-1)
            embed_normalized = torch.where(zero_mask[..., None],
                                           self.embedding.weight,
                                           embed_normalized)

            # Update embedding vectors with EMA
            norm_ema_inplace(self.embedding.weight, embed_normalized,
                             self.decay)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices
