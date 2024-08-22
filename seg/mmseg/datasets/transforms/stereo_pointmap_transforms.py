# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import os
import random
import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmcv.transforms import to_tensor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

Number = Union[int, float]

@TRANSFORMS.register_module()
class GenerateStereoPointmapCorrespondences(BaseTransform):
    def __init__(self, min_overlap_percentage=2, distance_threshold=0.1):
        self.min_overlap_percentage = min_overlap_percentage
        self.distance_threshold = distance_threshold
        return

    def transform(self, results: dict) -> dict:
        data_info = results['results1']
        other_data_info = results['results2']

        M1 = data_info['M'] ## 4 x 4
        M2 = other_data_info['M'] ## 4 x 4

        K1 = data_info['K'] ## 3 x 3
        K2 = other_data_info['K'] ## 3 x 3

        pointmap1 = data_info['gt_depth_map'].copy() ## 1024 x 768 x 3
        pointmap2 = other_data_info['gt_depth_map'].copy() ## 1024 x 768 x 3

        mask1 = data_info['mask']
        mask2 = other_data_info['mask']

        # Find correspondences
        pixel_coords1, pixel_coords2, overlap_percentage = self.find_correspondences(pointmap1, pointmap2, mask1, mask2, M1, M2, K1, K2)

        # Store the correspondences back into results
        results['results1']['pixel_coords1'] = pixel_coords1
        results['results1']['pixel_coords2'] = pixel_coords2
        results['results1']['overlap_percentage'] = overlap_percentage

        # ## debug
        # if pixel_coords1 is not None and pixel_coords2 is not None:
        #     image1 = data_info['img']
        #     image2 = other_data_info['img']
        #     combined_image = self.visualize_correspondences(image1, image2, pixel_coords1, pixel_coords2, vis_num_points=-1)
        #     seed = random.randint(0, 1000)
        #     cv2.imwrite('{}_image.jpg'.format(seed), combined_image); cv2.imwrite('{}_image1.jpg'.format(seed), image1); cv2.imwrite('{}_image2.jpg'.format(seed), image2)
        #     import ipdb; ipdb.set_trace()

        return results

    def __repr__(self):
        return self.__class__.__name__

    def find_correspondences(self, pointmap1, pointmap2, mask1, mask2, M1, M2, K1, K2):
        # Combine y and x indices to get pixel coordinates in image 1
        y_indices, x_indices = np.where(mask1 > 0)
        pixel_coords1 = np.vstack((x_indices, y_indices)).T

        points1 = pointmap1[mask1 > 0].reshape(-1, 3) ## N x 3

        # Convert points to homogeneous coordinates
        points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])

        # Transform points from camera 1 coordinates to camera 2 coordinates
        points_cam2 = M2 @ np.linalg.inv(M1) @ points1_h.T

        # Normalize homogeneous coordinates
        points_cam2 = points_cam2[:3, :] / points_cam2[3, :]

        # Project points to the image plane of camera 2
        points_image_plane2 = K2 @ points_cam2[:3, :]

        # Normalize image plane coordinates
        points_image_plane2 /= points_image_plane2[2, :]

        # Convert to pixel coordinates
        pixel_coords2 = points_image_plane2[:2, :].T
        image_height = pointmap1.shape[0]; image_width = pointmap1.shape[1]

        is_valid = (pixel_coords2[:, 0] >= 0) & \
                   (pixel_coords2[:, 0] < image_width) & \
                   (pixel_coords2[:, 1] >= 0) & \
                   (pixel_coords2[:, 1] < image_height)

        # Further check if the points are visible in the second camera's view
        pixel_coords2_valid = pixel_coords2[is_valid].astype(int)
        visible_points2 = pointmap2[pixel_coords2_valid[:, 1], pixel_coords2_valid[:, 0]]

        # Calculate depth in camera 2
        depth2_transformed = points_cam2[2, is_valid]  # Transformed depths of the valid points
        depth2 = visible_points2[:, 2]  # Actual depths in pointmap2

        # Check if the transformed depth is less than the depth in pointmap2
        is_visible = depth2_transformed <= depth2 + 1e-3

        # Compute Euclidean distance between transformed points and actual points
        distances = np.linalg.norm(points_cam2[:3, is_valid].T - visible_points2, axis=1)

        # Check if distances are within the threshold
        is_within_threshold = distances < self.distance_threshold

        # Combine visibility and distance threshold checks
        is_valid[is_valid] = is_visible & is_within_threshold

        overlap_percentage = (is_valid.sum() / (mask1 > 0).sum()) * 100

        if overlap_percentage < self.min_overlap_percentage:
            return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int), overlap_percentage

        pixel_coords1 = pixel_coords1[is_valid, :].astype(int)
        pixel_coords2 = pixel_coords2[is_valid, :].astype(int)

        return pixel_coords1, pixel_coords2, overlap_percentage

    def visualize_correspondences(self, image1, image2, pixel_coords1, pixel_coords2, vis_num_points=128):
        # Create a combined image by stacking image1 and image2 side by side
        height1, width1, _ = image1.shape
        height2, width2, _ = image2.shape
        combined_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
        combined_image[:height1, :width1, :] = image1
        combined_image[:height2, width1:width1 + width2, :] = image2

        # Randomly sample points
        if len(pixel_coords1) > vis_num_points and vis_num_points != -1:
            sampled_indices = np.random.choice(len(pixel_coords1), vis_num_points, replace=False)
            pixel_coords1 = pixel_coords1[sampled_indices]
            pixel_coords2 = pixel_coords2[sampled_indices]

        radius = 5
        # Draw points for corresponding coordinates
        for pt1, pt2 in zip(pixel_coords1, pixel_coords2):
            # Generate a random color
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Draw point in image1
            cv2.circle(combined_image, tuple(pt1), radius, color, -1)

            # Shift pt2 for the combined image
            pt2_shifted = (pt2[0] + width1, pt2[1])

            # Draw point in image2
            cv2.circle(combined_image, pt2_shifted, radius, color, -1)

        return combined_image

@TRANSFORMS.register_module()
class StereoPointmapTransformSecondaryToAnchor(BaseTransform):
    def __init__(self, background_val=-1000):
        self.background_val = background_val
        return

    def transform(self, results: dict) -> dict:
        data_info = results['results1']
        other_data_info = results['results2']

        # ## transform the point_map for other_data_info to the anchor coordinate system of data_info
        M1 = data_info['M']
        M2 = other_data_info['M']
        pointmap = other_data_info['gt_depth_map']

        pointmap_homogeneous = np.concatenate([pointmap, np.ones((pointmap.shape[0], pointmap.shape[1], 1))], axis=-1)
        pointmap_world = pointmap_homogeneous @ np.linalg.inv(M2).T
        pointmap_anchor = pointmap_world @ M1.T ## transform to the anchor camera coordinate system
        pointmap = pointmap_anchor[:, :, :3]  # Drop the homogeneous coordinate
        pointmap[other_data_info['mask'] == 0] = self.background_val ## set the background pixels to invalid

        other_data_info['gt_depth_map'] = pointmap
        results['results2'] = other_data_info

        # ##-----------------debug-------------------------
        # import open3d as o3d
        # mask1 = results['results1']['mask']
        # mask2 = results['results2']['mask']

        # pointmap1 = results['results1']['gt_depth_map'] ## 1024 x 768 x 3
        # pointmap2 = results['results2']['gt_depth_map'] ## 1024 x 768 x 3

        # pixel_coords1 = results['results1']['pixel_coords1'] ## num_pixels x 2
        # pixel_coords2 = results['results1']['pixel_coords2'] ## num_pixels x 2

        # points1 = pointmap1[mask1 > 0].reshape(-1, 3)
        # points2 = pointmap2[mask2 > 0].reshape(-1, 3)

        # pc1 = o3d.geometry.PointCloud()
        # pc2 = o3d.geometry.PointCloud()

        # pc1.points = o3d.utility.Vector3dVector(points1)
        # pc2.points = o3d.utility.Vector3dVector(points2)

        # # Set colors (blue for points1, red for points2)
        # pc1.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (points1.shape[0], 1)))  # Blue
        # pc2.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (points2.shape[0], 1)))  # Red

        # combined_pc = pc1 + pc2
        # seed = random.randint(0, 1000)
        # o3d.io.write_point_cloud("pc_{}.ply".format(seed), combined_pc)

        # ## visualize correspondences
        # if len(pixel_coords1) > 0 and len(pixel_coords2) > 0:
        #     points1 = pointmap1[pixel_coords1[:, 1], pixel_coords1[:, 0]] ## num_pixels x 3
        #     points2 = pointmap2[pixel_coords2[:, 1], pixel_coords2[:, 0]] ## num_pixels x 3

        #     pc1 = o3d.geometry.PointCloud()
        #     pc2 = o3d.geometry.PointCloud()

        #     pc1.points = o3d.utility.Vector3dVector(points1)
        #     pc2.points = o3d.utility.Vector3dVector(points2)

        #     pc1.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (points2.shape[0], 1)))
        #     pc2.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (points2.shape[0], 1)))

        #     combined_pc = pc1 + pc2
        #     o3d.io.write_point_cloud("corresp_pc_{}.ply".format(seed), combined_pc)

        #     diff = np.linalg.norm(points1 - points2, axis=1)
        #     print("seed:{} mean diff: {}, max diff:{}, min diff: {}".format(seed, np.mean(diff), np.max(diff), np.min(diff)))

        return results

    def __repr__(self):
        return self.__class__.__name__

@TRANSFORMS.register_module()
class PackStereoPointmapInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction',
                            'K', 'M')):
        self.meta_keys = meta_keys

    def transform(self, results: dict, min_depth=1e-2, max_depth=5) -> dict:
        packed_results = dict()
        results1 = results['results1']
        results2 = results['results2']

        if 'img' in results1:
            img = results1['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs1'] = img

        if 'img' in results2:
            img = results2['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs2'] = img

        data_sample1 = SegDataSample()
        data_sample2 = SegDataSample()

        ## stereo pointmap
        if 'gt_depth_map' in results1:
            ## clamp the depth to [min_depth, max_depth] for valid pixels
            gt_depth_map = results1['gt_depth_map'] ## 1024 x 768 x 3
            gt_depth_map = gt_depth_map.transpose(2, 0, 1) ## 3 x H x W, 3 x 1024 x 1024

            mask = results1['mask']
            gt_depth_map[2][mask > 0] = np.clip(gt_depth_map[2][mask > 0], min_depth, max_depth)

            gt_depth_data = dict(data=to_tensor(gt_depth_map.copy()))
            data_sample1.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        if 'gt_depth_map' in results2:
            ## clamp the depth to [min_depth, max_depth] for valid pixels
            gt_depth_map = results2['gt_depth_map'] ## 1024 x 768 x 3
            gt_depth_map = gt_depth_map.transpose(2, 0, 1) ## 3 x H x W, 3 x 1024 x 1024

            mask = results2['mask']
            gt_depth_map[2][mask > 0] = np.clip(gt_depth_map[2][mask > 0], min_depth, max_depth)

            gt_depth_data = dict(data=to_tensor(gt_depth_map.copy()))
            data_sample2.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta1 = {}
        img_meta2 = {}

        for key in self.meta_keys:
            if key in results1:
                img_meta1[key] = results1[key]

            if key in results2:
                img_meta2[key] = results2[key]

        data_sample1.set_metainfo(img_meta1)
        data_sample2.set_metainfo(img_meta2)

        packed_results['data_samples1'] = data_sample1
        packed_results['data_samples2'] = data_sample2

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class TestPackStereoPointmapInputs(BaseTransform):

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction',
                            'K', 'M')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        is_anchor = results['is_anchor']
        idx = 1
        if is_anchor == False:
            idx = 2

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs{}'.format(idx)] = img

        data_sample = SegDataSample()

        img_meta = {}

        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)

        packed_results['data_samples{}'.format(idx)] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
