# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch
import torch.nn.functional as F

## use cosine similarity
@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device='cpu', block_size=None, dist='dot'):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size)
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy()
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B


# @torch.no_grad()
# def bruteforce_nn_ratio(A, B, device='cpu', dist='dot'):
#     if isinstance(A, np.ndarray):
#         A = torch.from_numpy(A).to(device)
#     if isinstance(B, np.ndarray):
#         B = torch.from_numpy(B).to(device)

#     A = A.to(device)
#     B = B.to(device)

#     if dist == 'l2':
#         dist_func = torch.cdist
#         def top2(X, dim):
#             return torch.topk(X, k=2, dim=dim, largest=False) ## min for L2
#     elif dist == 'dot':
#         def dist_func(A, B):
#             return A @ B.T
#         def top2(X, dim):
#             return torch.topk(X, k=2, dim=dim, largest=True) ## max for dot
#     else:
#         raise ValueError(f'Unknown {dist=}')

#     dists = dist_func(A, B)
#     top2_A, nn_A = top2(dists, dim=1)
#     ratios_A = top2_A[:, 0] / (top2_A[:, 1] + 1e-8)  # Add small epsilon to avoid division by zero

#     nn_A = nn_A.cpu().numpy()
#     ratios_A = ratios_A.cpu().numpy()
#     return nn_A[:, 0], ratios_A  # Return the first nearest neighbor and the ratio of the two nearest neighbors


@torch.no_grad()
def bruteforce_nn_ratio(A, B, device='cpu', dist='dot', k=5):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        def topk(X, dim):
            return torch.topk(X, k=k, dim=dim, largest=False)  # min for L2
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T
        def topk(X, dim):
            return torch.topk(X, k=k, dim=dim, largest=True)  # max for dot
    else:
        raise ValueError(f'Unknown {dist=}')

    dists = dist_func(A, B)
    topk_A, nn_A = topk(dists, dim=1)
    
    # Calculate various ratios
    ratios_A = {
        'top1_to_top2': topk_A[:, 0] / (topk_A[:, 1] + 1e-8),
        'top1_to_mean_rest': topk_A[:, 0] / (torch.mean(topk_A[:, 1:], dim=1) + 1e-8),
        'top1_to_median_rest': topk_A[:, 0] / (torch.median(topk_A[:, 1:], dim=1)[0] + 1e-8),
        'top1_to_min_rest': topk_A[:, 0] / (torch.min(topk_A[:, 1:], dim=1)[0] + 1e-8),
    }

    nn_A = nn_A.cpu().numpy()
    ratios_A = {key: value.cpu().numpy() for key, value in ratios_A.items()}
    
    return nn_A[:, 0], ratios_A, topk_A.cpu().numpy()
