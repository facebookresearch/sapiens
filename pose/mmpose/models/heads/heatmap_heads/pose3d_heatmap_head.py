# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData, InstanceData
from torch import Tensor, nn
import numpy as np
from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead
from torch.nn import functional as F

OptIntSeq = Optional[Sequence[int]]

@MODELS.register_module()
class Pose3dHeatmapHead(BaseHead):
    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 num_keypoints: int,
                 image_to_heatmap_ratio: float = 4.0,
                 depth_conv_out_channels: OptIntSeq = (256, 256, 256),
                 depth_conv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 depth_final_layer: OptIntSeq = (1536, 1536, 1536),
                 K_conv_out_channels: OptIntSeq = (256, 256, 256),
                 K_conv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 K_final_layer: OptIntSeq = (1536, 1536, 1536),
                 loss: ConfigType = dict(type='KeypointMSELoss', use_target_weight=True),
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.image_to_heatmap_ratio = image_to_heatmap_ratio

        if isinstance(loss, dict):
            self.loss_module = MODELS.build(loss)
        elif isinstance(loss, (list, tuple)):
            self.loss_module = nn.ModuleList()
            for this_loss in loss:
                self.loss_module.append(MODELS.build(this_loss))
        else:
            raise TypeError(f'loss must be a dict or sequence of dict,\
                but got {type(loss)}')

        self.decoder = None

        self.depth_conv_layers = self._make_conv_layers(
            in_channels=in_channels,
            layer_out_channels=depth_conv_out_channels,
            layer_kernel_sizes=depth_conv_kernel_sizes)
        self.depth_final_layer = self._make_final_layer(depth_final_layer)

        self.K_conv_layers = self._make_conv_layers(
            in_channels=in_channels,
            layer_out_channels=K_conv_out_channels,
            layer_kernel_sizes=K_conv_kernel_sizes)
        self.K_final_layer = self._make_final_layer(K_final_layer)     

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_final_layer(self, final_layer: Sequence[int]) -> nn.Module:
        """Create final layer by given parameters."""
        layers = [nn.Flatten()]
        in_features = final_layer[0]

        for i in range(1, len(final_layer)):
            layers.append(nn.Linear(in_features, final_layer[i]))
            if i < len(final_layer) - 1:  # No activation after the last layer
                layers.append(nn.SiLU())
            in_features = final_layer[i]

        return nn.Sequential(*layers)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            stride = 2 # Set stride to 2 to reduce resolution by half
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))

            in_channels = out_channels

        return nn.Sequential(*layers)


    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Constant', layer='InstanceNorm2d', val=1, bias=0)  # Initialize gamma to 1 and beta to 0
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = feats[-1] ## B x C x H x W i.e 1 x 1536 x 64 x 48

        x_depth = self.depth_conv_layers(x)
        depth = self.depth_final_layer(x_depth) # B x num_keypoints

        x_K = self.K_conv_layers(x)
        K = self.K_final_layer(x_K) ## B x num_keypoints*4
        K = K.view(-1, self.num_keypoints, 4) ## B x num_keypoints x 4, fx, fy, cx, cy

        return (depth, K)

    def predict(self,
                feats: Features,
                batch_pred_instances: OptSampleList,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        (preds_depth, preds_K) = self.forward(feats)

        preds_pose2d = np.stack([pred.keypoints[0] for pred in batch_pred_instances], axis=0) ## B x num_keypoints x 2
        preds_pose2d = torch.from_numpy(preds_pose2d).to(preds_depth.device) ## B x num_keypoints x 

        preds_confidence = np.stack([pred.keypoint_scores[0] for pred in batch_pred_instances], axis=0) ## B x num_keypoints, numpy already

        preds_pose3d = self.compute_pose3d(depth=preds_depth, K=preds_K, pose2d=preds_pose2d) ## B x num_keypoints x 3
        
        preds_pose3d = to_numpy(preds_pose3d, unzip=True) #B x 308 x 3
        preds_K = to_numpy(preds_K, unzip=True) #B x 308 x 4
        preds_pose2d = to_numpy(preds_pose2d, unzip=True) #B x 308 x 2 

        ## important to add a dummy batch dimension to add_pred_to_data_sample
        preds = [
            InstanceData(keypoints_3d=pred_pose3d[np.newaxis, ...], \
                keypoints=pred_pose2d[np.newaxis, ...], \
                keypoint_scores=pred_confidence[np.newaxis, ...], \
                intrinsics=pred_K[np.newaxis, ...])
            for pred_pose3d, pred_pose2d, pred_K, pred_confidence in zip(preds_pose3d, preds_pose2d, preds_K, preds_confidence)
        ]

        return preds
    
    def compute_pose3d(self, depth: Tensor, K: Tensor, pose2d: Tensor) -> Tensor:
        """
        Compute 3D pose from 2D pose, depth, and per-keypoint camera intrinsics.

        Args:
            depth: Tensor of shape (B, N_keypoints) representing depth values for each keypoint
            K: Tensor of shape (B, N_keypoints, 4) representing camera intrinsics (fx, fy, cx, cy) per keypoint
            pose2d: Tensor of shape (B, N_keypoints, 2) representing 2D pose

        Returns:
            Tensor of shape (B, N_keypoints, 3) representing 3D pose
        """
        batch_size, num_keypoints, _ = pose2d.shape

        # Add shape assertions
        assert depth.shape == (batch_size, num_keypoints), "Depth shape mismatch"
        assert K.shape == (batch_size, num_keypoints, 4), "Camera intrinsics shape mismatch"
        assert pose2d.shape == (batch_size, num_keypoints, 2), "2D pose shape mismatch"

        # Unpack camera intrinsics
        fx = K[..., 0]  # Shape: (B, N_keypoints)
        fy = K[..., 1]  # Shape: (B, N_keypoints)
        cx = K[..., 2]  # Shape: (B, N_keypoints)
        cy = K[..., 3]  # Shape: (B, N_keypoints)

        # Extract 2D coordinates
        x = pose2d[..., 0]  # Shape: (B, N_keypoints)
        y = pose2d[..., 1]  # Shape: (B, N_keypoints)

        # Normalize coordinates
        x_norm = (x - cx) / fx
        y_norm = (y - cy) / fy

        # Create 3D coordinates
        z = depth  # Shape: (B, N_keypoints)
        x_3d = x_norm * z
        y_3d = y_norm * z

        # Combine into 3D coordinates
        pose3d = torch.stack([x_3d, y_3d, z], dim=-1)  # Shape: (B, N_keypoints, 3)

        return pose3d
    
    def compute_pose2d(self, pose3d: Tensor, K: Tensor) -> Tensor:
        """
        Compute 2D pose from 3D pose and per-keypoint camera intrinsics.

        Args:
            pose3d: Tensor of shape (B, N_keypoints, 3) representing 3D pose
            K: Tensor of shape (B, N_keypoints, 4) representing camera intrinsics (fx, fy, cx, cy) per keypoint

        Returns:
            Tensor of shape (B, N_keypoints, 2) representing 2D pose
        """
        batch_size, num_keypoints, _ = pose3d.shape

        # Add shape assertions
        assert pose3d.shape == (batch_size, num_keypoints, 3), "3D pose shape mismatch"
        assert K.shape == (batch_size, num_keypoints, 4), "Camera intrinsics shape mismatch"

        # Unpack camera intrinsics
        fx = K[..., 0]  # Shape: (B, N_keypoints)
        fy = K[..., 1]  # Shape: (B, N_keypoints)
        cx = K[..., 2]  # Shape: (B, N_keypoints)
        cy = K[..., 3]  # Shape: (B, N_keypoints)

        # Extract 3D coordinates
        X = pose3d[..., 0]  # Shape: (B, N_keypoints)
        Y = pose3d[..., 1]  # Shape: (B, N_keypoints)
        Z = pose3d[..., 2]  # Shape: (B, N_keypoints)

        # Project 3D points to 2D
        u = X * fx / Z + cx
        v = Y * fy / Z + cy

        pose2d = torch.stack([u, v], dim=-1)  # Shape: (B, N_keypoints, 2)

        return pose2d
    
    def decode_pose2d_heatmaps(self, heatmaps_original: torch.Tensor) -> Tensor:
        """Decode the heatmaps to keypoints using a differentiable approximation of argmax.
        Args:
            heatmaps (Tensor): B x K x H x W heatmaps with Gaussian targets.

        Returns:
            Tensor: The decoded keypoints in image coordinates (B x K x 2)
        """

        B, K, H, W = heatmaps_original.shape

        # Create a copy of heatmaps and detach it
        heatmaps = heatmaps_original.clone().detach()
    
        #  Create coordinate maps
        y_range = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device)
        x_range = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device)
        y_map, x_map = torch.meshgrid(y_range, x_range, indexing='ij')

        # Normalize heatmaps
        heatmaps_normalized = heatmaps / torch.sum(heatmaps.view(B, K, -1), dim=2).view(B, K, 1, 1)
        
        # Compute weighted sum of coordinates
        y_coord = torch.sum(heatmaps_normalized * y_map, dim=(2, 3))
        x_coord = torch.sum(heatmaps_normalized * x_map, dim=(2, 3))
        
        # Stack coordinates
        keypoints = torch.stack((x_coord, y_coord), dim=-1)
        
        # Scale keypoints to image coordinates
        keypoints = keypoints * self.image_to_heatmap_ratio
            
        return keypoints
    
    def decode_softargmax_pose2d_heatmap(self, heatmaps_original: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
        """Decode the heatmaps to keypoints using softargmax.
        Args:
        heatmaps_original (torch.Tensor): B x K x H x W heatmaps with Gaussian targets.
        beta (float): Temperature parameter for softmax. Higher values make it sharper.

        Returns:
            torch.Tensor: The decoded keypoints in image coordinates (B x K x 2)
        """
        B, K, H, W = heatmaps_original.shape

        # Create a copy of heatmaps and detach it
        heatmaps = heatmaps_original.clone().detach()

        # Create coordinate maps
        y_range = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device)
        x_range = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device)
        y_map, x_map = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Reshape heatmaps for softmax
        heatmaps_reshaped = heatmaps.view(B, K, -1)
        
        # Apply softmax with temperature
        heatmaps_softmax = F.softmax(beta * heatmaps_reshaped, dim=-1)
        
        # Reshape coordinate maps
        y_map_flat = y_map.reshape(-1)
        x_map_flat = x_map.reshape(-1)
        
        # Compute weighted sum of coordinates
        y_coord = torch.sum(heatmaps_softmax * y_map_flat, dim=-1)
        x_coord = torch.sum(heatmaps_softmax * x_map_flat, dim=-1)
        
        # Stack coordinates
        keypoints = torch.stack((x_coord, y_coord), dim=-1)
        
        # Scale keypoints to image coordinates
        keypoints = keypoints * self.image_to_heatmap_ratio
        
        return keypoints
    
    def decode_argmax_pose2d_heatmap(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Decode the heatmaps to keypoints using argmax.
        
        Args:
            heatmaps (torch.Tensor): B x K x H x W heatmaps with Gaussian targets.

        Returns:
            torch.Tensor: The decoded keypoints in image coordinates (B x K x 2)
        """
        B, K, H, W = heatmaps.shape

        # Flatten the last two dimensions
        heatmaps_flat = heatmaps.view(B, K, -1)
        
        # Get the indices of the maximum values
        max_indices = torch.argmax(heatmaps_flat, dim=-1)
        
        # Convert indices to y, x coordinates
        y_coord = max_indices // W
        x_coord = max_indices % W
        
        # Stack coordinates
        keypoints = torch.stack((x_coord, y_coord), dim=-1).float()
        
        # Scale keypoints to image coordinates
        keypoints = keypoints * self.image_to_heatmap_ratio
        
        return keypoints

    def loss(self,
             feats: Tuple[Tensor],
             pose2d_heatmaps: Tensor,
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:

        (pred_depth, pred_K) = self.forward(feats) ## pred_depth is B x num_keypoints, pred_K is B x num_keypoints x 4

        device = pred_depth.device
        gt_pose3d = torch.stack([torch.from_numpy(d.gt_instances.pose3d[0]).to(device) for d in batch_data_samples]) ## B x num_keypoints x 3
        gt_pose3d_visible = torch.stack([torch.from_numpy(d.gt_instances.pose3d_visible[0]).to(device) for d in batch_data_samples]) ## B x num_keypoints 
        
        ## B x num_keypoints x 2. do not use keypoints. use transformed_keypoints
        gt_pose2d = torch.stack([torch.from_numpy(d.gt_instances.transformed_keypoints[0].astype(np.float32)).to(device) for d in batch_data_samples]) 

        ## filtered according to transformed keypoints
        gt_pose2d_visible = torch.stack([torch.from_numpy(d.gt_instances.keypoints_visible[0]).to(device) for d in batch_data_samples]) ## B x num_keypoints. 

        ## remove out of bound keypoints of gt_pose2d_visible
        gt_K = torch.stack([torch.from_numpy(d.K.astype(np.float32)).to(device) for d in batch_data_samples]) ##
        gt_fx = gt_K[:, 0, 0]; gt_fy = gt_K[:, 1, 1]; gt_cx = gt_K[:, 0, 2]; gt_cy = gt_K[:, 1, 2]
        gt_K = torch.stack([gt_fx, gt_fy, gt_cx, gt_cy], dim=1) ## B x 4
        gt_K = gt_K.view(-1, 1, 4).repeat(1, self.num_keypoints, 1) ## B x num_keypoints x 4. copy per keypoint

        ## compute 2d pose coordinates from the downsamples heatmaps. we clone and detach the heatmaps to avoid backprop
        # pred_pose2d = self.decode_pose2d_heatmaps(pose2d_heatmaps) ## B x num_keypoints x 2 ## gaussian expectation
        pred_pose2d = self.decode_softargmax_pose2d_heatmap(pose2d_heatmaps) ## B x num_keypoints x 2 ## soft argmax with beta=100
        # hard_pred_pose2d = self.decode_argmax_pose2d_heatmap(pose2d_heatmaps) ## B x num_keypoints x 2 ## hard argmax

        ## compute pred_pose2d using pred_pose3d, pred_K
        pred_pose3d = self.compute_pose3d(depth=pred_depth, K=pred_K, pose2d=pred_pose2d) ## B x num_keypoints x 3

        # calculate losses
        losses = self.compute_losses(pred_pose3d, gt_pose3d, gt_pose3d_visible, pred_pose2d, gt_pose2d, gt_pose2d_visible, pred_K, gt_K)

        return losses, {'pose3d': pred_pose3d, 'K': pred_K, 'pose2d': pred_pose2d} 
    
    def compute_losses(self, pred_pose3d, gt_pose3d, gt_pose3d_visible, pred_pose2d, gt_pose2d, gt_pose2d_visible, pred_K, gt_K):
        """
        pred_pose3d and gt_pose3d: B x N_keypoints x 3
        gt_pose3d_visible: B x N_keypoints
        pred_pose2d and gt_pose2d: B x N_keypoints x 2
        gt_pose2d_visible: B x N_keypoints
        pred_K and gt_K: B x N_keypoints x 4. fx, fy, cx, cy
        pred_confidence: B x N_keypoints
        """
        losses = dict()
        if not isinstance(self.loss_module, nn.ModuleList):
            losses_decode = [self.loss_module]
        else:
            losses_decode = self.loss_module

        for loss_decode in losses_decode:
            
            ## loss on X, Y, Z. computed using pred_depth, pred_K, gt_pose2d
            if loss_decode.loss_name == 'loss_pose3d_l1':
                this_loss = loss_decode(
                            output=pred_pose3d,
                            target=gt_pose3d,
                            target_weight=gt_pose3d_visible,
                            )
            
            ## loss on fx, fy, cx, cy
            elif loss_decode.loss_name == 'loss_pose3d_K':
                this_loss = loss_decode(
                            output=pred_K,
                            target=gt_K,
                            target_weight=gt_pose3d_visible,
                            )
            ## on relative depth Z
            elif loss_decode.loss_name == 'loss_pose3d_rel_depth':
                this_loss = loss_decode(
                            output=pred_pose3d[:, :, 2],
                            target=gt_pose3d[:, :, 2],
                            target_weight=gt_pose3d_visible,
                            )
            
            elif loss_decode.loss_name == 'loss_pose3d_pose2d_l1':
                this_loss = loss_decode(
                            output=pred_pose2d,
                            target=gt_pose2d,
                            target_weight=gt_pose2d_visible,
                            )
            elif loss_decode.loss_name == 'loss_pose3d_depth_l1':
                this_loss = loss_decode(
                            output=pred_pose3d[:, :, 2],
                            target=gt_pose3d[:, :, 2],
                            target_weight=gt_pose3d_visible,
                            )
            
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = this_loss
            else:
                losses[loss_decode.loss_name] += this_loss

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DeepposeRegressionHead` (before MMPose v1.0.0) to a
        compatible format of :class:`RegressionHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v
    
    
