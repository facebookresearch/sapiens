import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from mmengine.model import MMDistributedDataParallel

import matplotlib.pyplot as plt
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D

from torch.nn import functional as F
import numpy as np
import os
import cv2
import tempfile
import matplotlib
import torch

try:
    import open3d as o3d
except Exception as e:
    warnings.warn('Cannot import open3d!')

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

##-----------------------------------------------------------------------------------------------------------
@HOOKS.register_module()
class StereoCorrespondencesVisualizationHook(Hook):
    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 max_samples = 4,
                 max_pixels = 24000,
                 vis_image_width = 768,
                 vis_image_height = 512,
                 backend_args: Optional[dict] = None):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw

        self.max_samples = max_samples
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height
        self.max_pixels = max_pixels
        return

    def after_train_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        # ## check if the rank is 0
        if not runner.rank == 0:
            return

        total_curr_iter = runner.iter

        if total_curr_iter % self.interval != 0:
            return

        inputs1 = data_batch['inputs1'] ## list of images as tensor. BGR images. they are made RGB by model preprocessor
        inputs2 = data_batch['inputs2'] ## list of images as tensor. BGR images. they are made RGB by model preprocessor

        data_samples1 = data_batch['data_samples1']
        data_samples2 = data_batch['data_samples2']

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis_data')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(total_curr_iter).zfill(6)

        suffix += '_' + data_samples1[0].img_path.split('/')[-1].replace('.png', '')
        suffix += '_' + data_samples2[0].img_path.split('/')[-1].replace('.png', '')

        batch_size = min(self.max_samples, len(inputs1))

        inputs1 = inputs1[:batch_size]
        inputs2 = inputs2[:batch_size]

        input_image_height = inputs1[0].shape[1]
        input_image_width = inputs1[0].shape[2]

        data_samples1 = data_samples1[:batch_size]
        data_samples2 = data_samples2[:batch_size]

        if isinstance(runner.model, MMDistributedDataParallel):
            ## check the distributed.py in mmengine.
            ## this triggers the mmpretrain condition with vis_masks
            descs1 = outputs['vis_preds'][:batch_size] ## B x 32 x H x W
            descs2 = outputs['vis_masks'][:batch_size] ## B x 32 x H x W
        else:
            descs1 = outputs['vis_preds'][0][:batch_size] ## B x 32 x H x W
            descs2 = outputs['vis_preds'][1][:batch_size] ## B x 32 x H x W

        # Check if the model is wrapped with MMDistributedDataParallel
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        data_samples_with_pred1 = model.postprocess_train_result(descs1, data_samples1) ##
        data_samples_with_pred2 = model.postprocess_train_result(descs2, data_samples2) ##

        descs1 = descs1.cpu().detach() ## B x 32 x 512 x 384
        descs1 = F.interpolate(descs1, size=(input_image_height, input_image_width), mode='bilinear', align_corners=False) ## B x 32 x 1024 x 768
        descs1 = descs1.numpy().transpose((0, 2, 3, 1)) ## B x H x W x 32

        descs2 = descs2.cpu().detach() ## B x 32 x 512 x 384
        descs2 = F.interpolate(descs2, size=(input_image_height, input_image_width), mode='bilinear', align_corners=False) ## B x 32 x 1024 x 768
        descs2 = descs2.numpy().transpose((0, 2, 3, 1)) ## B x H x W x 32

        vis_images = []

        for i, (input1, input2, data_sample_with_pred1, desc1, data_sample_with_pred2, desc2) in \
            enumerate(zip(inputs1, inputs2, data_samples_with_pred1, descs1, data_samples_with_pred2, descs2)):
            image1 = input1.permute(1, 2, 0).cpu().numpy() ## bgr image
            image1 = np.ascontiguousarray(image1.copy())

            image2 = input2.permute(1, 2, 0).cpu().numpy() ## bgr image
            image2 = np.ascontiguousarray(image2.copy())

            image = np.concatenate([image1, image2], axis=1)

            ## gt correspondences
            gt_pixel_coords1 =  data_sample_with_pred1.pixel_coords1 ## num_pixels x 2
            gt_pixel_coords2 =  data_sample_with_pred1.pixel_coords2 ## num_pixels x 2

            if len(gt_pixel_coords1) > self.max_pixels:
                gt_pixel_coords1 = gt_pixel_coords1[:self.max_pixels]
                gt_pixel_coords2 = gt_pixel_coords2[:self.max_pixels]

            mask1 = data_sample_with_pred1.mask
            mask2 = data_sample_with_pred2.mask

            vis_gt_correspondences = self.visualize_correspondences(image1, image2, gt_pixel_coords1, gt_pixel_coords2) ## visualize gt correspondences
            pred_pixel_coords2 = self.get_pred_matching(desc1, desc2, mask1, mask2, gt_pixel_coords1)
            vis_pred_correspondences = self.visualize_correspondences(image1, image2, gt_pixel_coords1, pred_pixel_coords2)  ## visualize pred correspondences 
            vis_image = np.concatenate([vis_gt_correspondences, vis_pred_correspondences], axis=1)
            vis_image = cv2.resize(vis_image, (2 * self.vis_image_width, self.vis_image_height), interpolation=cv2.INTER_AREA)
            vis_images.append(vis_image)

        grid_image = np.concatenate(vis_images, axis=0)

        # Save the grid image to a file
        grid_out_file = '{}_{}.jpg'.format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

        return
    
    def get_pred_matching(self, desc1, desc2, mask1, mask2, gt_pixel_coords1):
        # Extract descriptors at gt_pixel_coords1
        gt_descriptors1 = desc1[gt_pixel_coords1[:, 1], gt_pixel_coords1[:, 0], :]  # num_pixels x 32

        # Filter desc2 using mask2
        masked_indices = np.nonzero(mask2)
        desc2_filtered = desc2[masked_indices]  # num human pixels x 32

        # Use bruteforce reciprocal nearest neighbor matching
        nn_A, nn_B = bruteforce_reciprocal_nns(gt_descriptors1, desc2_filtered)
        
        # Convert matched indices to 2D coordinates in image2
        matched_coords2 = np.column_stack((masked_indices[1][nn_A], masked_indices[0][nn_A]))
        
        return matched_coords2

    def visualize_correspondences(self, image1, image2, pixel_coords1, pixel_coords2):
        # Concatenate images
        image = np.concatenate([image1, image2], axis=1)
        
        # Make sure the points are integers
        pixel_coords1 = pixel_coords1.astype(int)
        pixel_coords2 = pixel_coords2.astype(int)
        
        # Offset the coordinates of image2 by the width of image1 for correct alignment in the concatenated image
        pixel_coords2[:, 0] += image1.shape[1]
        
        # Create a copy of the concatenated image to draw on
        image_with_correspondences = image.copy()
        
        # Get the jet colormap
        color_map = matplotlib.cm.jet
        
        num_points = pixel_coords1.shape[0]
        
        for i in range(num_points):
            # Get color for the point
            color = np.array(color_map(i / max(1, float(num_points - 1)))[:3]) * 255
            color = tuple(map(int, color))
            
            # Draw points
            cv2.circle(image_with_correspondences, tuple(pixel_coords1[i]), 3, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image_with_correspondences, tuple(pixel_coords2[i]), 3, color, -1, lineType=cv2.LINE_AA)
            
        return image_with_correspondences
