# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.image import imflip
from mmengine import is_seq_of
from scipy.stats import truncnorm
from mmpose.structures.bbox import bbox_xyxy2cs, flip_bbox
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmpose.registry import TRANSFORMS
from mmpose.structures.keypoint import flip_keypoints_custom_center
from mmpose.structures.keypoint import flip_keypoints
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix
from .formatting import PackPoseInputs
from mmpose.utils.typing import MultiConfig
from scipy.stats import norm


@TRANSFORMS.register_module()
class Pose3dGenerateTarget(BaseTransform):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def transform(self, results: Dict) -> Optional[dict]:
        if 'keypoints_depth' not in results:
            num_keypoints = results['transformed_keypoints'].shape[1]
            results['pose3d'] = np.zeros((num_keypoints, 3)).astype(np.float32)
            results['pose3d_visible'] = np.zeros(num_keypoints, dtype=bool)
            results['K'] = np.eye(3).astype(np.float32)
            return results
        
        assert 'K' in results
        results['K'] = results['K'].astype(np.float32)

        K = results['K']
        height, width = results['img'].shape[:2]

        keypoints = results['transformed_keypoints'][0] ## 308 x 2
        keypoints_valid = results['keypoints_visible'][0] ## 308. this is actually filtered using transformed keypoints

        Z = results['keypoints_depth'][0, :, 0] ## 308 

        # Compute X, Y, Z as pose3d
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        X = (keypoints[:, 0] - cx) * Z / fx
        Y = (keypoints[:, 1] - cy) * Z / fy

        # Stack X, Y, Z to create pose3d
        pose3d = np.stack([X, Y, Z], axis=-1)
        pose2d = np.dot(K, pose3d.T).T  ## project 3d keypoints to 2D
        pose2d = pose2d[:, :2] / (pose2d[:, 2:] + 1e-8) 

        keypoints_valid = keypoints_valid * (pose2d[:, 0] >= 0) * (
                    pose2d[:, 0] < width) * (pose2d[:, 1] >= 0) * (pose2d[:, 1] < height)

        # Apply validity mask
        pose3d[keypoints_valid == 0] = 0
        pose2d[keypoints_valid == 0] = 0

        results['pose3d'] = pose3d.astype(np.float32)
        results['pose3d_visible'] = keypoints_valid.astype(bool)

        # # ## debug
        # image = results['img']

        # # Draw only visible keypoints
        # for i in range(len(keypoints)):
        #     u = int(pose2d[i, 0])
        #     v = int(pose2d[i, 1])
        #     if keypoints_valid[i] and u >= 0 and u < image.shape[1] \
        #             and v >= 0 and v < image.shape[0]:
                
        #         # Projected keypoint in red
        #         cv2.circle(image, (u, v), 3, (0, 255, 0), -1)

        # # Save debug image
        # random_seed = np.random.randint(0, 100000)
        # cv2.imwrite('pose3d_{}.png'.format(random_seed), image)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class PackPose3dInputs(PackPoseInputs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        ## this condition is used for inference only
        if 'keypoints_visible' not in results:
            return super().transform(results)

        ## clean up keypoints_visible for out of bound keypoints
        is_visible = results['keypoints_visible'] ## 1 x N_keypoints
        image_width, image_height = results['input_size']
        transformed_keypoints = results['transformed_keypoints'] ## 1 x N_keypoints x 2

        is_visible = is_visible * (transformed_keypoints[:, :, 0] >= 0) * (transformed_keypoints[:, :, 0] < image_width) \
                        * (transformed_keypoints[:, :, 1] >= 0) * (transformed_keypoints[:, :, 1] < image_height)

        results['keypoints_visible'] = is_visible

        ## zero out out of bound keypoints
        results['transformed_keypoints'][is_visible == 0] = 0

        packed_results = super().transform(results) ## call packposeinputs

        ## TODO: this if condition should not be needed but still crashes. investigate.
        if 'pose3d' not in results:
            num_keypoints = results['transformed_keypoints'].shape[1] ## keypoints is 1 x N x 2
            results['pose3d'] = np.zeros((num_keypoints, 3)).astype(np.float32)
            results['pose3d_visible'] = np.zeros(num_keypoints, dtype=bool)
            results['K'] = np.eye(3).astype(np.float32)

        packed_results['data_samples'].gt_instances.set_field(results['pose3d'].reshape(1, -1, 3), 'pose3d') ## 1 x N_keypoints x 3
        packed_results['data_samples'].gt_instances.set_field(results['pose3d_visible'].reshape(1, -1), 'pose3d_visible') ## 1 x N_keypoints

        if 'depth_heatmap' in results:
            packed_results['data_samples'].gt_instances.set_field(results['depth_heatmap'][np.newaxis, ...], 'depth_heatmap') ## 1 x N_keypoints x num_bins

        return packed_results

@TRANSFORMS.register_module()
class Pose3dRandomBBoxTransform(BaseTransform):
    def __init__(self,
                 shift_factor: float = 0.16,
                 shift_prob: float = 0.3,
                 scale_factor: Tuple[float, float] = (0.5, 1.5),
                 scale_prob: float = 1.0,) -> None:
        super().__init__()

        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob

    @staticmethod
    def _truncnorm(low: float = -1.,
                   high: float = 1.,
                   size: tuple = ()) -> np.ndarray:
        """Sample from a truncated normal distribution."""
        return truncnorm.rvs(low, high, size=size).astype(np.float32)

    @cache_randomness
    def _get_transform_params(self, num_bboxes: int) -> Tuple:
        """Get random transform parameters.

        Args:
            num_bboxes (int): The number of bboxes

        Returns:
            tuple:
            - offset (np.ndarray): Offset factor of each bbox in shape (n, 2)
            - scale (np.ndarray): Scaling factor of each bbox in shape (n, 1)
            - rotate (np.ndarray): Rotation degree of each bbox in shape (n,)
        """
        # Get shift parameters
        offset = self._truncnorm(size=(num_bboxes, 2)) * self.shift_factor
        offset = np.where(
            np.random.rand(num_bboxes, 1) < self.shift_prob, offset, 0.)

        # Get scaling parameters
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = self._truncnorm(size=(num_bboxes, 1)) * sigma + mu
        scale = np.where(
            np.random.rand(num_bboxes, 1) < self.scale_prob, scale, 1.)

        return offset, scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`RandomBboxTransform`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        bbox_scale = results['bbox_scale']
        num_bboxes = bbox_scale.shape[0]

        offset, scale = self._get_transform_params(num_bboxes)

        results['bbox_center'] += offset * bbox_scale
        results['bbox_scale'] *= scale

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(shift_prob={self.shift_prob}, '
        repr_str += f'shift_factor={self.shift_factor}, '
        repr_str += f'scale_prob={self.scale_prob}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        return repr_str

@TRANSFORMS.register_module()
class Pose3dRandomFlip(BaseTransform):
    def __init__(self,
                 prob: Union[float, List[float]] = 0.5,
                 direction: Union[str, List[str]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      List) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def transform(self, results: dict) -> dict:
        flip_dir = self._choose_direction()
        img_shape = results['img'].shape[:2] ## 1024 x 768

        if flip_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = flip_dir

            h, w = results.get('input_size', results['img_shape'])
            # flip image and mask
            if isinstance(results['img'], list):
                results['img'] = [
                    imflip(img, direction=flip_dir) for img in results['img']
                ]
            else:
                results['img'] = imflip(results['img'], direction=flip_dir)

            if 'img_mask' in results:
                results['img_mask'] = imflip(
                    results['img_mask'], direction=flip_dir)

            # flip bboxes
            if results.get('bbox', None) is not None:
                results['bbox'] = flip_bbox(
                    results['bbox'],
                    image_size=(w, h),
                    bbox_format='xyxy',
                    direction=flip_dir)

            if results.get('bbox_center', None) is not None:
                results['bbox_center'] = flip_bbox(
                    results['bbox_center'],
                    image_size=(w, h),
                    bbox_format='center',
                    direction=flip_dir)

            # flip keypoints
            if results.get('keypoints', None) is not None:
                keypoints, keypoints_visible = flip_keypoints(
                    results['keypoints'],
                    results.get('keypoints_visible', None),
                    image_size=(w, h),
                    flip_indices=results['flip_indices'],
                    direction=flip_dir)

                results['keypoints'] = keypoints
                results['keypoints_visible'] = keypoints_visible
            
            # flip camera parameters
            if 'K' in results.keys():
                # Flip the principal point for the left-right flipped image
                results['K'][0, 2] = img_shape[1] - results['K'][0, 2] - 1
            
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'
        return repr_str

@TRANSFORMS.register_module()
class RandomFlipAroundRoot(BaseTransform):
    """Data augmentation with random horizontal joint flip around a root joint.

    Args:
        keypoints_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``keypoints``. Please
            refer to the docstring of the ``flip_keypoints_custom_center``
            function for more details.
        target_flip_cfg (dict): Configurations of the
            ``flip_keypoints_custom_center`` function for ``lifting_target``.
            Please refer to the docstring of the
            ``flip_keypoints_custom_center`` function for more details.
        flip_prob (float): Probability of flip. Default: 0.5.
        flip_camera (bool): Whether to flip horizontal distortion coefficients.
            Default: ``False``.

    Required keys:
        keypoints
        lifting_target

    Modified keys:
        (keypoints, keypoints_visible, lifting_target, lifting_target_visible,
        camera_param)
    """

    def __init__(self,
                 keypoints_flip_cfg,
                 target_flip_cfg,
                 flip_prob=0.5,
                 flip_camera=False):
        self.keypoints_flip_cfg = keypoints_flip_cfg
        self.target_flip_cfg = target_flip_cfg
        self.flip_prob = flip_prob
        self.flip_camera = flip_camera

    def transform(self, results: Dict) -> dict:
        """The transform function of :class:`ZeroCenterPose`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        keypoints = results['keypoints']
        if 'keypoints_visible' in results:
            keypoints_visible = results['keypoints_visible']
        else:
            keypoints_visible = np.ones(keypoints.shape[:-1], dtype=np.float32)
        lifting_target = results['lifting_target']
        if 'lifting_target_visible' in results:
            lifting_target_visible = results['lifting_target_visible']
        else:
            lifting_target_visible = np.ones(
                lifting_target.shape[:-1], dtype=np.float32)

        if np.random.rand() <= self.flip_prob:
            if 'flip_indices' not in results:
                flip_indices = list(range(self.num_keypoints))
            else:
                flip_indices = results['flip_indices']

            # flip joint coordinates
            keypoints, keypoints_visible = flip_keypoints_custom_center(
                keypoints, keypoints_visible, flip_indices,
                **self.keypoints_flip_cfg)
            lifting_target, lifting_target_visible = flip_keypoints_custom_center(  # noqa
                lifting_target, lifting_target_visible, flip_indices,
                **self.target_flip_cfg)

            results['keypoints'] = keypoints
            results['keypoints_visible'] = keypoints_visible
            results['lifting_target'] = lifting_target
            results['lifting_target_visible'] = lifting_target_visible

            # flip horizontal distortion coefficients
            if self.flip_camera:
                assert 'camera_param' in results, \
                    'Camera parameters are missing.'
                _camera_param = deepcopy(results['camera_param'])

                assert 'c' in _camera_param
                _camera_param['c'][0] *= -1

                if 'p' in _camera_param:
                    _camera_param['p'][0] *= -1

                results['camera_param'].update(_camera_param)

        return results

@TRANSFORMS.register_module()
class Pose3dTopdownAffine(BaseTransform):
    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        w, h = self.input_size
        warp_size = (int(w), int(h)) # (width, height), 768 x 1024

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        rot = 0. ## no rotation

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)

        ## convert K from entire image to the cropped image
        if 'K' in results:
            K = results['K']

            # Adjust focal lengths based on the scale factors
            translation_x = center[0] - scale[0] / 2
            translation_y = center[1] - scale[1] / 2
            scale_factor_x = w / scale[0]
            scale_factor_y = h / scale[1]
            
            c_x_new = (K[0, 2] - translation_x) * scale_factor_x
            c_y_new = (K[1, 2] - translation_y) * scale_factor_y
            
            f_x_new = K[0, 0] * scale_factor_x
            f_y_new = K[1, 1] * scale_factor_y

            # Update the intrinsic matrix
            results['K'] = np.array([
                [f_x_new, 0, c_x_new],
                [0, f_y_new, c_y_new],
                [0, 0, 1]
            ])

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str
