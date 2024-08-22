# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) ByteDance, Inc. and its affiliates. All rights reserved.
# Modified from https://github.com/keyu-tian/SparK/blob/main/encoder.py
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


class SparseHelper:
    """The helper to compute sparse operation with pytorch, such as sparse
    convlolution, sparse batch norm, etc."""

    _cur_active: torch.Tensor = None

    @staticmethod
    def _get_active_map_or_index(H: int,
                                 returning_active_map: bool = True
                                 ) -> torch.Tensor:
        """Get current active map with (B, 1, f, f) shape or index format."""
        # _cur_active with shape (B, 1, f, f)
        downsample_raito = H // SparseHelper._cur_active.shape[-1]
        active_ex = SparseHelper._cur_active.repeat_interleave(
            downsample_raito, 2).repeat_interleave(downsample_raito, 3)
        return active_ex if returning_active_map else active_ex.squeeze(
            1).nonzero(as_tuple=True)

    @staticmethod
    def sp_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse convolution forward function."""
        x = super(type(self), self).forward(x)

        # (b, c, h, w) *= (b, 1, h, w), mask the output of conv
        x *= SparseHelper._get_active_map_or_index(
            H=x.shape[2], returning_active_map=True)
        return x

    @staticmethod
    def sp_bn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse batch norm forward function."""
        active_index = SparseHelper._get_active_map_or_index(
            H=x.shape[2], returning_active_map=False)

        # (b, c, h, w) -> (b, h, w, c)
        x_permuted = x.permute(0, 2, 3, 1)

        # select the features on non-masked positions to form flatten features
        # with shape (n, c)
        x_flattened = x_permuted[active_index]

        # use BN1d to normalize this flatten feature (n, c)
        x_flattened = super(type(self), self).forward(x_flattened)

        # generate output
        output = torch.zeros_like(x_permuted, dtype=x_flattened.dtype)
        output[active_index] = x_flattened

        # (b, h, w, c) -> (b, c, h, w)
        output = output.permute(0, 3, 1, 2)
        return output


class SparseConv2d(nn.Conv2d):
    """hack: override the forward function.
    See `sp_conv_forward` above for more details
    """
    forward = SparseHelper.sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    """hack: override the forward function.
    See `sp_conv_forward` above for more details
    """
    forward = SparseHelper.sp_conv_forward


class SparseAvgPooling(nn.AvgPool2d):
    """hack: override the forward function.
    See `sp_conv_forward` above for more details
    """
    forward = SparseHelper.sp_conv_forward


@MODELS.register_module()
class SparseBatchNorm2d(nn.BatchNorm1d):
    """hack: override the forward function.
    See `sp_bn_forward` above for more details
    """
    forward = SparseHelper.sp_bn_forward


@MODELS.register_module()
class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    """hack: override the forward function.
    See `sp_bn_forward` above for more details
    """
    forward = SparseHelper.sp_bn_forward


@MODELS.register_module('SparseLN2d')
class SparseLayerNorm2D(nn.LayerNorm):
    """Implementation of sparse LayerNorm on channels for 2d images."""

    def forward(self,
                x: torch.Tensor,
                data_format='channel_first') -> torch.Tensor:
        """Sparse layer norm forward function with 2D data.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, (
            f'LayerNorm2d only supports inputs with shape '
            f'(N, C, H, W), but got tensor with shape {x.shape}')
        if data_format == 'channel_last':
            index = SparseHelper._get_active_map_or_index(
                H=x.shape[1], returning_active_map=False)

            # select the features on non-masked positions to form flatten
            # features with shape (n, c)
            x_flattened = x[index]
            # use LayerNorm to normalize this flatten feature (n, c)
            x_flattened = super().forward(x_flattened)

            # generate output
            x = torch.zeros_like(x, dtype=x_flattened.dtype)
            x[index] = x_flattened
        elif data_format == 'channel_first':
            index = SparseHelper._get_active_map_or_index(
                H=x.shape[2], returning_active_map=False)
            x_permuted = x.permute(0, 2, 3, 1)

            # select the features on non-masked positions to form flatten
            # features with shape (n, c)
            x_flattened = x_permuted[index]
            # use LayerNorm to normalize this flatten feature (n, c)
            x_flattened = super().forward(x_flattened)

            # generate output
            x = torch.zeros_like(x_permuted, dtype=x_flattened.dtype)
            x[index] = x_flattened
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise NotImplementedError
        return x
