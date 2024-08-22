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
import matplotlib.cm as cm
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os
import cv2
import tempfile
try:
    import open3d as o3d
except Exception as e:
    warnings.warn('Cannot import open3d!')

##-----------------------------------------------------------------------------------------------------------
@HOOKS.register_module()
class StereoPointmapVisualizationHook(Hook):
    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 max_samples = 4,
                 vis_image_width = 512,
                 vis_image_height = 1024,
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

    def vis_depth_map(self, point_map, mask=None, background_color=100):
        depth_map = point_map[:, :, 2] ### x,y,z. z is the depth
        if mask is None:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = 1 - depth_map ## 1 is near camera, 0 is far camera
            depth_map = (depth_map * 255.0).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
            return depth_colored

        depth_foreground = depth_map[mask > 0] ## value in range [0, 1]
        processed_depth = np.full((mask.shape[0], mask.shape[1], 3), background_color, dtype=np.uint8)

        if len(depth_foreground) == 0:
            return processed_depth

        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val)) ## 1 is near camera, 0 is far camera
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)

        depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
        depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)

        processed_depth[mask] = depth_colored_foreground

        return processed_depth

    def vis_stereo_point_map(self, point_map1, point_map2, mask1=None, mask2=None):
        H, W = point_map1.shape[:2]

        if mask1.sum() == 0 or mask2.sum() == 0:
            return np.zeros((H, W, 3), dtype=np.float32)

        # Flatten the point maps and masks
        points1 = point_map1[mask1].reshape(-1, 3)
        points2 = point_map2[mask2].reshape(-1, 3)

        # Create a figure for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot first half point cloud
        ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='b', s=1, depthshade=True)

        # Plot second half point cloud
        ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='r', s=1, depthshade=True)

        # Set the viewpoint
        ax.view_init(elev=0, azim=0)  # Adjust elevation and azimuth for better camera-like view

        # Set axes to equal for xyz
        max_range = np.array([points1[:, 0].max()-points1[:, 0].min(),
                            points1[:, 1].max()-points1[:, 1].min(),
                            points1[:, 2].max()-points1[:, 2].min()]).max() / 2.0
        mid_x = (points1[:, 0].max()+points1[:, 0].min()) * 0.5
        mid_y = (points1[:, 1].max()+points1[:, 1].min()) * 0.5
        mid_z = (points1[:, 2].max()+points1[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Hide the axes
        ax.set_axis_off()

        # Use a temporary file to save the plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.savefig(tmp_file.name, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the figure to free memory

        # Read the image using OpenCV
        img_bgr = cv2.imread(tmp_file.name, cv2.IMREAD_COLOR)

        # Delete the temporary file
        os.remove(tmp_file.name)

        return img_bgr

    def get_point_cloud(self, point_map1, point_map2, mask1=None, mask2=None):
        H, W = point_map1.shape[:2]

        if mask1.sum() == 0 or mask2.sum() == 0:
            return None

        # Flatten the point maps and masks
        points1 = point_map1[mask1].reshape(-1, 3)
        points2 = point_map2[mask2].reshape(-1, 3)

        # # Create Open3D point cloud for points1 and color them blue
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        blue_color = np.array([0, 0, 1])  # Blue
        pcd1.colors = o3d.utility.Vector3dVector(np.tile(blue_color, (len(points1), 1)))

        # Create Open3D point cloud for points2 and color them red
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        red_color = np.array([1, 0, 0])  # Red
        pcd2.colors = o3d.utility.Vector3dVector(np.tile(red_color, (len(points2), 1)))

        # Combine the point clouds
        combined_pcd = pcd1 + pcd2

        return combined_pcd

    def after_train_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """

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

        data_samples1 = data_samples1[:batch_size]
        data_samples2 = data_samples2[:batch_size]

        if isinstance(runner.model, MMDistributedDataParallel):
            ## check the distributed.py in mmengine.
            ## this triggers the mmpretrain condition with vis_masks
            seg_logits1 = outputs['vis_preds'][:batch_size] ## B x 3 x H x W
            seg_logits2 = outputs['vis_masks'][:batch_size] ## B x 3 x H x W
        else:
            seg_logits1 = outputs['vis_preds'][0][:batch_size] ## B x 3 x H x W
            seg_logits2 = outputs['vis_preds'][1][:batch_size] ## B x 3 x H x W

        # Check if the model is wrapped with MMDistributedDataParallel
        model = runner.model.module if isinstance(runner.model, MMDistributedDataParallel) else runner.model
        data_samples_with_pred1 = model.postprocess_train_result(seg_logits1, data_samples1) ##
        data_samples_with_pred2 = model.postprocess_train_result(seg_logits2, data_samples2) ##

        seg_logits1 = seg_logits1.cpu().detach().numpy() ## B x 3 x H x W
        seg_logits1 = seg_logits1.transpose((0, 2, 3, 1)) ## B x H x W x 3

        seg_logits2 = seg_logits2.cpu().detach().numpy() ## B x 3 x H x W
        seg_logits2 = seg_logits2.transpose((0, 2, 3, 1)) ## B x H x W x 3

        vis_images = []

        for i, (input1, input2, data_sample_with_pred1, seg_logit1, data_sample_with_pred2, seg_logit2) in \
            enumerate(zip(inputs1, inputs2, data_samples_with_pred1, seg_logits1, data_samples_with_pred2, seg_logits2)):
            image1 = input1.permute(1, 2, 0).cpu().numpy() ## bgr image
            image1 = np.ascontiguousarray(image1.copy())

            image2 = input2.permute(1, 2, 0).cpu().numpy() ## bgr image
            image2 = np.ascontiguousarray(image2.copy())

            image = np.concatenate([image1, image2], axis=1)

            K1 = data_sample_with_pred1.K ## intrinsics 3 x 3
            K2 = data_sample_with_pred2.K ## intrinsics 3 x 3

            gt_pointmap1 = data_sample_with_pred1.gt_depth_map.data.numpy() ## 3 x H x W
            gt_pointmap1 = gt_pointmap1.transpose((1, 2, 0)) ## H x W x 3

            gt_pointmap2 = data_sample_with_pred2.gt_depth_map.data.numpy() ## 3 x H x W
            gt_pointmap2 = gt_pointmap2.transpose((1, 2, 0)) ## H x W x 3

            mask1 = gt_pointmap1[:, :, 0] > -100  # Assuming the first channel indicates valid points
            mask2 = gt_pointmap2[:, :, 0] > -100  # Assuming the first channel indicates valid points

            vis_gt_pointmap = self.vis_stereo_point_map(gt_pointmap1, gt_pointmap2, mask1, mask2)
            vis_gt_pointmap = cv2.resize(vis_gt_pointmap, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

            ## resize the pointmaps
            seg_logit1_X = seg_logit1[:, :, 0] ## H x W
            seg_logit1_Y = seg_logit1[:, :, 1] ## H x W
            seg_logit1_Z = seg_logit1[:, :, 2] ## H x W

            seg_logit2_X = seg_logit2[:, :, 0] ## H x W
            seg_logit2_Y = seg_logit2[:, :, 1] ## H x W
            seg_logit2_Z = seg_logit2[:, :, 2] ## H x W

            seg_logit1_X_resized = cv2.resize(seg_logit1_X, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
            seg_logit1_Y_resized = cv2.resize(seg_logit1_Y, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
            seg_logit1_Z_resized = cv2.resize(seg_logit1_Z, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)

            seg_logit2_X_resized = cv2.resize(seg_logit2_X, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
            seg_logit2_Y_resized = cv2.resize(seg_logit2_Y, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
            seg_logit2_Z_resized = cv2.resize(seg_logit2_Z, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Combine resized channels back into one array
            pred_pointmap1 = np.stack((seg_logit1_X_resized, seg_logit1_Y_resized, seg_logit1_Z_resized), axis=-1)
            pred_pointmap2 = np.stack((seg_logit2_X_resized, seg_logit2_Y_resized, seg_logit2_Z_resized), axis=-1)

            # Resize prediction to the size of the image
            vis_pred_pointmap = self.vis_stereo_point_map(pred_pointmap1, pred_pointmap2, mask1, mask2)
            vis_pred_pointmap = cv2.resize(vis_pred_pointmap, (image1.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

            ## visualize anchor pointmaps as depth
            vis_gt_depth = self.vis_depth_map(gt_pointmap1, mask1)
            vis_gt_depth = cv2.resize(vis_gt_depth, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

            vis_pred_depth = self.vis_depth_map(pred_pointmap1, mask1)
            vis_pred_depth = cv2.resize(vis_pred_depth, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

            ## visualize secondary as depth
            vis_gt_depth2 = self.vis_depth_map(gt_pointmap2, mask2)
            vis_gt_depth2 = cv2.resize(vis_gt_depth2, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)

            vis_pred_depth2 = self.vis_depth_map(pred_pointmap2, mask2)
            vis_pred_depth2 = cv2.resize(vis_pred_depth2, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)

            # vis_image = np.concatenate([image, vis_gt_pointmap, vis_gt_depth, vis_pred_pointmap, vis_pred_depth], axis=1)
            vis_image = np.concatenate([image, vis_gt_pointmap, vis_gt_depth, vis_gt_depth2, vis_pred_pointmap, vis_pred_depth, vis_pred_depth2], axis=1)

            vis_image = cv2.resize(vis_image, (3 * self.vis_image_width, self.vis_image_height), interpolation=cv2.INTER_AREA)
            vis_images.append(vis_image)

            ### save the point cloud
            # gt_pc = self.get_point_cloud(gt_pointmap1, gt_pointmap2, mask1, mask2)
            # if gt_pc is not None:
            #     ply_out_file = '{}_{}_{}.ply'.format(prefix, suffix, i)
            #     o3d.io.write_point_cloud(ply_out_file, gt_pc)

        grid_image = np.concatenate(vis_images, axis=0)

        # Save the grid image to a file
        grid_out_file = '{}_{}.jpg'.format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

        return
