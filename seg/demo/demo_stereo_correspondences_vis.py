from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import stereo_pointmap_inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import cv2
import numpy as np
import tempfile
import torchvision
import torch
import matplotlib
from mmseg.utils import bruteforce_reciprocal_nns, bruteforce_nn_ratio

torchvision.disable_beta_transforms_warning()

## default
# MIN_RATIO_THRESHOLDS={
#     'top1_to_top2': 1.02,
#     'top1_to_mean_rest': 1.1,
#     'top1_to_min_rest': 1.1,
# }

## most relaxed
MIN_RATIO_THRESHOLDS={
    'top1_to_top2': 0,
    'top1_to_mean_rest': 0,
    'top1_to_min_rest': 0,
}


## device = cuda or cpu
def visualize_correspondences(image1, image2, desc1, desc2, mask1, mask2, device='cpu', downsample_factor=4):
    # Convert boolean masks to uint8
    mask1_uint8 = mask1.astype(np.uint8) * 255
    mask2_uint8 = mask2.astype(np.uint8) * 255

    # Downsample masks
    mask1_downsampled = cv2.resize(mask1_uint8, (mask1.shape[1] // downsample_factor, mask1.shape[0] // downsample_factor), interpolation=cv2.INTER_NEAREST)
    mask2_downsampled = cv2.resize(mask2_uint8, (mask2.shape[1] // downsample_factor, mask2.shape[0] // downsample_factor), interpolation=cv2.INTER_NEAREST)

    # Convert back to boolean
    mask1_downsampled = mask1_downsampled > 0
    mask2_downsampled = mask2_downsampled > 0

    # Extract descriptors at downsampled coordinates
    gt_pixel_coords1_downsampled = np.argwhere(mask1_downsampled)  # (y, x) order
    gt_descriptors1 = desc1[gt_pixel_coords1_downsampled[:, 0] * downsample_factor, 
                            gt_pixel_coords1_downsampled[:, 1] * downsample_factor, :]  # num_pixels x 32

    # Filter desc2 using downsampled mask2
    masked_indices_downsampled = np.nonzero(mask2_downsampled)  # Returns (y_indices, x_indices)
    desc2_filtered = desc2[masked_indices_downsampled[0] * downsample_factor, 
                           masked_indices_downsampled[1] * downsample_factor, :]  # num human pixels x 32

    # Use bruteforce reciprocal nearest neighbor matching
    nn_indices, ratios, topk_values = bruteforce_nn_ratio(gt_descriptors1, desc2_filtered, device='cuda', dist='dot', k=5)
    good_matches_1 = ratios['top1_to_top2'] > MIN_RATIO_THRESHOLDS['top1_to_top2']
    good_matches_2 = ratios['top1_to_mean_rest'] > MIN_RATIO_THRESHOLDS['top1_to_mean_rest']
    good_matches_3 = ratios['top1_to_min_rest'] > MIN_RATIO_THRESHOLDS['top1_to_min_rest']

    # print('top1_to_top2 max:{} min:{}'.format(ratios['top1_to_top2'].max(), ratios['top1_to_top2'].min()))
    # print('top1_to_mean_rest max:{} min:{}'.format(ratios['top1_to_mean_rest'].max(), ratios['top1_to_mean_rest'].min()))
    # print('top1_to_min_rest max:{} min:{}'.format(ratios['top1_to_min_rest'].max(), ratios['top1_to_min_rest'].min()))

    # Combine strategies
    good_matches = good_matches_1 & good_matches_2 & good_matches_3 

    # Convert matched indices to 2D coordinates in image2
    matched_coords2 = np.column_stack((masked_indices_downsampled[1][nn_indices], masked_indices_downsampled[0][nn_indices]))  # (x, y) order

    matched_coords1 = gt_pixel_coords1_downsampled * downsample_factor  # Scale up to original image size
    matched_coords2 = matched_coords2 * downsample_factor  # Scale back up to original image size

    # Concatenate images
    image = np.concatenate([image1, image2], axis=1)

    # Offset the coordinates of image2 by the width of image1 for correct alignment in the concatenated image
    matched_coords2[:, 0] += image1.shape[1]

    # Create a copy of the concatenated image to draw on
    image_with_correspondences = image.copy()

    # Get the jet colormap
    color_map = matplotlib.cm.jet

    num_points = gt_pixel_coords1_downsampled.shape[0]

    for i in range(num_points):
        # Get color for the point
        color = np.array(color_map(i / max(1, float(num_points - 1)))[:3]) * 255
        color = tuple(map(int, color))

        # Draw points
        pt1 = tuple(matched_coords1[i][::-1].astype(int))  # Swap x and y for OpenCV        
        cv2.circle(image_with_correspondences, pt1, 3, color, -1, lineType=cv2.LINE_AA)
        
        if good_matches[i] == True:
            pt2 = tuple(matched_coords2[i].astype(int))
            cv2.circle(image_with_correspondences, pt2, 3, color, -1, lineType=cv2.LINE_AA)

    return image_with_correspondences

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument('--seg_dir', '--seg-dir', default=None, help='Path to segmentation dir')
    parser.add_argument('--anchor_frame_idx', default=0,)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

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

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    seg_dir = args.seg_dir
    anchor_frame_idx = int(args.anchor_frame_idx)
    anchor_frame_idx = anchor_frame_idx if anchor_frame_idx < len(image_names) else anchor_frame_idx % len(image_names)

    extension = image_names[0].split('.')[-1]
    global_image_names = sorted([x for x in os.listdir(input_dir) if x.endswith('.' + extension)])

    model_input_size = model.data_preprocessor.size ## height x width

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_name1 = global_image_names[anchor_frame_idx]
        image_name2 = image_name

        # print('anchor image: {} vs {}'.format(image_name1, image_name2))

        image_path1 = os.path.join(input_dir, image_name1)
        image_path2 = os.path.join(input_dir, image_name2)

        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        mask_path1 = os.path.join(seg_dir, image_name1.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        mask_path2 = os.path.join(seg_dir, image_name2.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))

        mask1 = np.load(mask_path1)
        mask2 = np.load(mask_path2)

        mask1 = mask1.astype(np.uint8) * 255
        mask2 = mask2.astype(np.uint8) * 255

        ### resize to input resolution of vision transformer
        # image1 = cv2.resize(image1, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_AREA)
        # image2 = cv2.resize(image2, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_AREA)
        # mask1 = cv2.resize(mask1, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_NEAREST)
        # mask2 = cv2.resize(mask2, (model_input_size[1], model_input_size[0]), interpolation=cv2.INTER_NEAREST)
        
        result = stereo_pointmap_inference_model(model, image1, image2)

        result1 = result.pred_desc1.data.cpu().numpy()
        result2 = result.pred_desc2.data.cpu().numpy()

        desc1 = result1.transpose(1, 2, 0) ### (H, W, C). per pixel 32 dim descriptor
        desc2 = result2.transpose(1, 2, 0) ### (H, W, C). per pixel 32 dim descriptor

        ##----------------------------------------------------
        vis_image_correspondences = visualize_correspondences(image1, image2, desc1, desc2, mask1, mask2, device='cpu', downsample_factor=8)
        # vis_image_correspondences = visualize_correspondences(image1, image2, desc1, desc2, mask1, mask2, device='cpu', downsample_factor=1)
        # vis_image_correspondences = visualize_correspondences(image1, image2, desc1, desc2, mask1, mask2, device='cpu', downsample_factor=4)

        ##----------------------------------------------------
        extension = image_name1.split('.')[-1]
        extension = '.' + extension
        save_image_name = '{}_{}{}'.format(image_name1.replace(extension, ''), image_name2.replace(extension, ''), extension)
        output_file = os.path.join(args.output_root, save_image_name)

        vis_image = np.concatenate([image1, image2, vis_image_correspondences], axis=1)
        cv2.imwrite(output_file, vis_image)

if __name__ == '__main__':
    main()
