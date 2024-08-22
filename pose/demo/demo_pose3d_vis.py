import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from tqdm import tqdm
import warnings
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import pickle

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='mmengine')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.functional')
warnings.filterwarnings("ignore", category=UserWarning, module='json_tricks.encoders')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def process_one_image(args,
                      img,
                      image_height,
                      image_width,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    # predict 2D keypoints and 3D keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes=None) ## bboxes is None, so it will use the whole image
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)

def draw_keypoints_3d(visualizer, image, keypoints_3d, keypoints_3d_scores, kpt_thr, xy_axis_limit, z_axis_limit):

    ## 270 for elevation and 90 for azimuth is working
    vis_image= visualizer._draw_3d_keypoints(
            image,
            keypoints_3d,
            keypoints_3d_scores,
            kpt_thr = kpt_thr,
            num_instances = -1,
            axis_elev = 270,  
            axis_azimuth = 90, 
            axis_roll=0,
            x_axis_limit = xy_axis_limit,
            y_axis_limit = xy_axis_limit,
            z_axis_limit = z_axis_limit,
            axis_dist = 1.0, ## this has no effect. TO DO: check
            radius = 100,
            thickness = 10,
        )

    side_vis_image = visualizer._draw_3d_keypoints(
            image,
            keypoints_3d,
            keypoints_3d_scores,
            kpt_thr = kpt_thr,
            num_instances = -1,
            axis_elev = 0,    # Elevation: 0 degrees (looking straight ahead)
            axis_azimuth = -90, # Azimuth: -90 degrees (facing the YZ plane)
            axis_roll = 0,    # Roll: 0 degrees (no rotation of the view)
            x_axis_limit = xy_axis_limit,
            y_axis_limit = xy_axis_limit,
            z_axis_limit = z_axis_limit,
            axis_dist = 1.0, ## this has no effect. TO DO: check
            radius = 100,
            thickness = 10,
        )

    vis_image = np.concatenate((vis_image, side_vis_image), axis=1)
    return vis_image

def save_keypoints_as_ply(keypoints_3d, keypoints_3d_scores, colors, output_path, threshold=0.5):
    import open3d as o3d

    colors_normalized = colors.astype(float) / 255.0

    # Create a list to store all geometries
    geometries = []

    # Iterate over each person
    for person_keypoints, person_scores in zip(keypoints_3d, keypoints_3d_scores):
        # Create spheres for each keypoint of the person
        for i, (point, score) in enumerate(zip(person_keypoints, person_scores)):
            if score > 0:  # Only create spheres for visible keypoints
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(point)
                
                # Use the normalized color corresponding to this keypoint index
                color = colors_normalized[i]
                sphere.paint_uniform_color(color)
                
                geometries.append(sphere)

    # Add a blue sphere at the origin to represent the camera
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    camera_sphere.paint_uniform_color([0, 0, 1])  # Blue color
    geometries.append(camera_sphere)

    # Combine all geometries
    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in geometries:
        combined_mesh += geom

    # Save the PLY file
    o3d.io.write_triangle_mesh(output_path, combined_mesh)
    return

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument('--xy_axis_limit', type=float, default=1.6, help='XY limit: [-val/2, val/2]')
    parser.add_argument('--z_axis_limit', type=float, default=2.0, help='Z limit: [-val/2, val/2]')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True, # dont load the checkpoint meta data, load from config file
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    visualizer_3d = VISUALIZERS.build(pose_estimator.cfg.visualizer_3d)
    visualizer_3d.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    ## no skeleton vis for 3d pose.
    # visualizer_3d.skeleton = None

    ##--------debug------------
    # ## brown, orange, green, purple for 13, 15, 16, 17
    # visualizer_3d.kpt_color[13] = [165, 42, 42]
    # visualizer_3d.kpt_color[15] = [255, 165, 0]
    # visualizer_3d.kpt_color[16] = [0, 128, 0]
    # visualizer_3d.kpt_color[17] = [128, 0, 128]

    # ## pink, blue, red, yellow for 14, 18, 19, 20
    # visualizer_3d.kpt_color[14] = [255, 20, 147]
    # visualizer_3d.kpt_color[18] = [0, 0, 255]
    # visualizer_3d.kpt_color[19] = [255, 0, 0]
    # visualizer_3d.kpt_color[20] = [255, 255, 0]

    ##---------debug-------------
    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)  # Join the directory path with the image file name
        image = cv2.imread(image_path)

        pred_instances = process_one_image(args, image_path, image.shape[0], image.shape[1], pose_estimator, visualizer)
        img_vis = mmcv.rgb2bgr(visualizer.get_image())

        output_file = os.path.join(args.output_root, os.path.basename(image_path))

        ## draw 3D keypoints
        keypoints_3d = pred_instances.keypoints_3d ## P x 308 x 3
        keypoints_3d_scores = pred_instances.keypoint_scores ## P x 308 ## copy pose2d scores
        keypoints = pred_instances.keypoints ## P x 308 x 2

        # save_keypoints_as_ply(keypoints_3d, keypoints_3d_scores, visualizer_3d.kpt_color, 'pose3d.ply')
        # import ipdb; ipdb.set_trace()

        img_vis_pose_3d = draw_keypoints_3d(visualizer_3d, image, keypoints_3d, keypoints_3d_scores, \
                    kpt_thr=args.kpt_thr, xy_axis_limit=args.xy_axis_limit, z_axis_limit = args.z_axis_limit)

        img_vis = np.concatenate((img_vis, img_vis_pose_3d), axis=1) ## pose2d + pose3d
        # img_vis = np.concatenate((image, img_vis_pose_3d), axis=1) ## image + pose3d
        mmcv.imwrite(img_vis, output_file)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)
            pred_save_path = os.path.join(output_file.replace('.jpg', '.json').replace('.png', '.json'))

            with open(pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')

if __name__ == '__main__':
    main()
