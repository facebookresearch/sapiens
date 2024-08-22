import torch
import torch.utils.data
import torch.multiprocessing as mp
import numpy as np
import os
import cv2
from PIL import ImageDraw
from tqdm import tqdm
import io
import json
import copy
from PIL import Image
from care.data.io import typed
from concurrent.futures import ThreadPoolExecutor
import random
from matplotlib import pyplot as plt
from xtcocotools.coco import COCO

GOLIATH_KP_ORDER = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_big_toe_tip',
    'left_small_toe_tip',
    'left_heel',
    'right_big_toe_tip',
    'right_small_toe_tip',
    'right_heel',
    'right_thumb_tip',
    'right_thumb_first_joint',
    'right_thumb_second_joint',
    'right_thumb_third_joint',
    'right_index_tip',
    'right_index_first_joint',
    'right_index_second_joint',
    'right_index_third_joint',
    'right_middle_tip',
    'right_middle_first_joint',
    'right_middle_second_joint',
    'right_middle_third_joint',
    'right_ring_tip',
    'right_ring_first_joint',
    'right_ring_second_joint',
    'right_ring_third_joint',
    'right_pinky_tip',
    'right_pinky_first_joint',
    'right_pinky_second_joint',
    'right_pinky_third_joint',
    'right_wrist',
    'left_thumb_tip',
    'left_thumb_first_joint',
    'left_thumb_second_joint',
    'left_thumb_third_joint',
    'left_index_tip',
    'left_index_first_joint',
    'left_index_second_joint',
    'left_index_third_joint',
    'left_middle_tip',
    'left_middle_first_joint',
    'left_middle_second_joint',
    'left_middle_third_joint',
    'left_ring_tip',
    'left_ring_first_joint',
    'left_ring_second_joint',
    'left_ring_third_joint',
    'left_pinky_tip',
    'left_pinky_first_joint',
    'left_pinky_second_joint',
    'left_pinky_third_joint',
    'left_wrist',
    'left_olecranon',
    'right_olecranon',
    'left_cubital_fossa',
    'right_cubital_fossa',
    'left_acromion',
    'right_acromion',
    'neck',
    'center_of_glabella',
    'center_of_nose_root',
    'tip_of_nose_bridge',
    'midpoint_1_of_nose_bridge',
    'midpoint_2_of_nose_bridge',
    'midpoint_3_of_nose_bridge',
    'center_of_labiomental_groove',
    'tip_of_chin',
    'upper_startpoint_of_r_eyebrow',
    'lower_startpoint_of_r_eyebrow',
    'end_of_r_eyebrow',
    'upper_midpoint_1_of_r_eyebrow',
    'lower_midpoint_1_of_r_eyebrow',
    'upper_midpoint_2_of_r_eyebrow',
    'upper_midpoint_3_of_r_eyebrow',
    'lower_midpoint_2_of_r_eyebrow',
    'lower_midpoint_3_of_r_eyebrow',
    'upper_startpoint_of_l_eyebrow',
    'lower_startpoint_of_l_eyebrow',
    'end_of_l_eyebrow',
    'upper_midpoint_1_of_l_eyebrow',
    'lower_midpoint_1_of_l_eyebrow',
    'upper_midpoint_2_of_l_eyebrow',
    'upper_midpoint_3_of_l_eyebrow',
    'lower_midpoint_2_of_l_eyebrow',
    'lower_midpoint_3_of_l_eyebrow',
    'l_inner_end_of_upper_lash_line',
    'l_outer_end_of_upper_lash_line',
    'l_centerpoint_of_upper_lash_line',
    'l_midpoint_2_of_upper_lash_line',
    'l_midpoint_1_of_upper_lash_line',
    'l_midpoint_6_of_upper_lash_line',
    'l_midpoint_5_of_upper_lash_line',
    'l_midpoint_4_of_upper_lash_line',
    'l_midpoint_3_of_upper_lash_line',
    'l_outer_end_of_upper_eyelid_line',
    'l_midpoint_6_of_upper_eyelid_line',
    'l_midpoint_2_of_upper_eyelid_line',
    'l_midpoint_5_of_upper_eyelid_line',
    'l_centerpoint_of_upper_eyelid_line',
    'l_midpoint_4_of_upper_eyelid_line',
    'l_midpoint_1_of_upper_eyelid_line',
    'l_midpoint_3_of_upper_eyelid_line',
    'l_midpoint_6_of_upper_crease_line',
    'l_midpoint_2_of_upper_crease_line',
    'l_midpoint_5_of_upper_crease_line',
    'l_centerpoint_of_upper_crease_line',
    'l_midpoint_4_of_upper_crease_line',
    'l_midpoint_1_of_upper_crease_line',
    'l_midpoint_3_of_upper_crease_line',
    'r_inner_end_of_upper_lash_line',
    'r_outer_end_of_upper_lash_line',
    'r_centerpoint_of_upper_lash_line',
    'r_midpoint_1_of_upper_lash_line',
    'r_midpoint_2_of_upper_lash_line',
    'r_midpoint_3_of_upper_lash_line',
    'r_midpoint_4_of_upper_lash_line',
    'r_midpoint_5_of_upper_lash_line',
    'r_midpoint_6_of_upper_lash_line',
    'r_outer_end_of_upper_eyelid_line',
    'r_midpoint_3_of_upper_eyelid_line',
    'r_midpoint_1_of_upper_eyelid_line',
    'r_midpoint_4_of_upper_eyelid_line',
    'r_centerpoint_of_upper_eyelid_line',
    'r_midpoint_5_of_upper_eyelid_line',
    'r_midpoint_2_of_upper_eyelid_line',
    'r_midpoint_6_of_upper_eyelid_line',
    'r_midpoint_3_of_upper_crease_line',
    'r_midpoint_1_of_upper_crease_line',
    'r_midpoint_4_of_upper_crease_line',
    'r_centerpoint_of_upper_crease_line',
    'r_midpoint_5_of_upper_crease_line',
    'r_midpoint_2_of_upper_crease_line',
    'r_midpoint_6_of_upper_crease_line',
    'l_inner_end_of_lower_lash_line',
    'l_outer_end_of_lower_lash_line',
    'l_centerpoint_of_lower_lash_line',
    'l_midpoint_2_of_lower_lash_line',
    'l_midpoint_1_of_lower_lash_line',
    'l_midpoint_6_of_lower_lash_line',
    'l_midpoint_5_of_lower_lash_line',
    'l_midpoint_4_of_lower_lash_line',
    'l_midpoint_3_of_lower_lash_line',
    'l_outer_end_of_lower_eyelid_line',
    'l_midpoint_6_of_lower_eyelid_line',
    'l_midpoint_2_of_lower_eyelid_line',
    'l_midpoint_5_of_lower_eyelid_line',
    'l_centerpoint_of_lower_eyelid_line',
    'l_midpoint_4_of_lower_eyelid_line',
    'l_midpoint_1_of_lower_eyelid_line',
    'l_midpoint_3_of_lower_eyelid_line',
    'r_inner_end_of_lower_lash_line',
    'r_outer_end_of_lower_lash_line',
    'r_centerpoint_of_lower_lash_line',
    'r_midpoint_1_of_lower_lash_line',
    'r_midpoint_2_of_lower_lash_line',
    'r_midpoint_3_of_lower_lash_line',
    'r_midpoint_4_of_lower_lash_line',
    'r_midpoint_5_of_lower_lash_line',
    'r_midpoint_6_of_lower_lash_line',
    'r_outer_end_of_lower_eyelid_line',
    'r_midpoint_3_of_lower_eyelid_line',
    'r_midpoint_1_of_lower_eyelid_line',
    'r_midpoint_4_of_lower_eyelid_line',
    'r_centerpoint_of_lower_eyelid_line',
    'r_midpoint_5_of_lower_eyelid_line',
    'r_midpoint_2_of_lower_eyelid_line',
    'r_midpoint_6_of_lower_eyelid_line',
    'tip_of_nose',
    'bottom_center_of_nose',
    'r_outer_corner_of_nose',
    'l_outer_corner_of_nose',
    'inner_corner_of_r_nostril',
    'outer_corner_of_r_nostril',
    'upper_corner_of_r_nostril',
    'inner_corner_of_l_nostril',
    'outer_corner_of_l_nostril',
    'upper_corner_of_l_nostril',
    'r_outer_corner_of_mouth',
    'l_outer_corner_of_mouth',
    'center_of_cupid_bow',
    'center_of_lower_outer_lip',
    'midpoint_1_of_upper_outer_lip',
    'midpoint_2_of_upper_outer_lip',
    'midpoint_1_of_lower_outer_lip',
    'midpoint_2_of_lower_outer_lip',
    'midpoint_3_of_upper_outer_lip',
    'midpoint_4_of_upper_outer_lip',
    'midpoint_5_of_upper_outer_lip',
    'midpoint_6_of_upper_outer_lip',
    'midpoint_3_of_lower_outer_lip',
    'midpoint_4_of_lower_outer_lip',
    'midpoint_5_of_lower_outer_lip',
    'midpoint_6_of_lower_outer_lip',
    'r_inner_corner_of_mouth',
    'l_inner_corner_of_mouth',
    'center_of_upper_inner_lip',
    'center_of_lower_inner_lip',
    'midpoint_1_of_upper_inner_lip',
    'midpoint_2_of_upper_inner_lip',
    'midpoint_1_of_lower_inner_lip',
    'midpoint_2_of_lower_inner_lip',
    'midpoint_3_of_upper_inner_lip',
    'midpoint_4_of_upper_inner_lip',
    'midpoint_5_of_upper_inner_lip',
    'midpoint_6_of_upper_inner_lip',
    'midpoint_3_of_lower_inner_lip',
    'midpoint_4_of_lower_inner_lip',
    'midpoint_5_of_lower_inner_lip',
    'midpoint_6_of_lower_inner_lip',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'teeth',
    'l_top_end_of_inferior_crus',
    'l_top_end_of_superior_crus',
    'l_start_of_antihelix',
    'l_end_of_antihelix',
    'l_midpoint_1_of_antihelix',
    'l_midpoint_1_of_inferior_crus',
    'l_midpoint_2_of_antihelix',
    'l_midpoint_3_of_antihelix',
    'l_point_1_of_inner_helix',
    'l_point_2_of_inner_helix',
    'l_point_3_of_inner_helix',
    'l_point_4_of_inner_helix',
    'l_point_5_of_inner_helix',
    'l_point_6_of_inner_helix',
    'l_point_7_of_inner_helix',
    'l_highest_point_of_antitragus',
    'l_bottom_point_of_tragus',
    'l_protruding_point_of_tragus',
    'l_top_point_of_tragus',
    'l_start_point_of_crus_of_helix',
    'l_deepest_point_of_concha',
    'l_tip_of_ear_lobe',
    'l_midpoint_between_22_15',
    'l_bottom_connecting_point_of_ear_lobe',
    'l_top_connecting_point_of_helix',
    'l_point_8_of_inner_helix',
    'r_top_end_of_inferior_crus',
    'r_top_end_of_superior_crus',
    'r_start_of_antihelix',
    'r_end_of_antihelix',
    'r_midpoint_1_of_antihelix',
    'r_midpoint_1_of_inferior_crus',
    'r_midpoint_2_of_antihelix',
    'r_midpoint_3_of_antihelix',
    'r_point_1_of_inner_helix',
    'r_point_8_of_inner_helix',
    'r_point_3_of_inner_helix',
    'r_point_4_of_inner_helix',
    'r_point_5_of_inner_helix',
    'r_point_6_of_inner_helix',
    'r_point_7_of_inner_helix',
    'r_highest_point_of_antitragus',
    'r_bottom_point_of_tragus',
    'r_protruding_point_of_tragus',
    'r_top_point_of_tragus',
    'r_start_point_of_crus_of_helix',
    'r_deepest_point_of_concha',
    'r_tip_of_ear_lobe',
    'r_midpoint_between_22_15',
    'r_bottom_connecting_point_of_ear_lobe',
    'r_top_connecting_point_of_helix',
    'r_point_2_of_inner_helix',
    'l_center_of_iris',
    'l_border_of_iris_3',
    'l_border_of_iris_midpoint_1',
    'l_border_of_iris_12',
    'l_border_of_iris_midpoint_4',
    'l_border_of_iris_9',
    'l_border_of_iris_midpoint_3',
    'l_border_of_iris_6',
    'l_border_of_iris_midpoint_2',
    'r_center_of_iris',
    'r_border_of_iris_3',
    'r_border_of_iris_midpoint_1',
    'r_border_of_iris_12',
    'r_border_of_iris_midpoint_4',
    'r_border_of_iris_9',
    'r_border_of_iris_midpoint_3',
    'r_border_of_iris_6',
    'r_border_of_iris_midpoint_2',
    'l_center_of_pupil',
    'l_border_of_pupil_3',
    'l_border_of_pupil_midpoint_1',
    'l_border_of_pupil_12',
    'l_border_of_pupil_midpoint_4',
    'l_border_of_pupil_9',
    'l_border_of_pupil_midpoint_3',
    'l_border_of_pupil_6',
    'l_border_of_pupil_midpoint_2',
    'r_center_of_pupil',
    'r_border_of_pupil_3',
    'r_border_of_pupil_midpoint_1',
    'r_border_of_pupil_12',
    'r_border_of_pupil_midpoint_4',
    'r_border_of_pupil_9',
    'r_border_of_pupil_midpoint_3',
    'r_border_of_pupil_6',
    'r_border_of_pupil_midpoint_2',
]

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

GOLIATH_KP_CONNECTIONS = kp_connections(GOLIATH_KP_ORDER)

# ------------------------------------------------------------------------------------
def goliath_vis_keypoints(image, kps, vis_thres=0.3, alpha=0.7):
    # image is [image_size, image_size, RGB] #numpy array
    # kps is [17, 3] #numpy array
    kps = kps.astype(np.int16)
    bgr_image = image[:, :, ::-1] ##if this is directly in function call, this produces weird opecv cv2 Umat errors
    kp_image = vis_keypoints(bgr_image, kps.T, vis_thres, alpha) #convert to bgr
    kp_image = kp_image[:, :, ::-1] #bgr to rgb

    return kp_image

# ------------------------------------------------------------------------------------
def vis_keypoints(img, kps, kp_thresh=-1, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (3, #keypoints) where 3 rows are (x, y, depth z).
    needs a BGR image as it only uses opencv functions, returns a bgr image
    """
    dataset_keypoints = GOLIATH_KP_ORDER
    kp_lines = GOLIATH_KP_CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) // 2
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) // 2
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')

    ## Draw all the keypoints in white color circle, radius = 4
    for kp in kps.T:
        if kp[2] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, (int(kp[0]), int(kp[1])),
                radius=4, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        kp_mask = cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            kp_mask = cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=3, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p1,
                radius=4, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            kp_mask = cv2.circle(
                kp_mask, p2,
                radius=4, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    ## weird opencv bug on cv2UMat vs numpy
    if type(kp_mask) != type(img):
        kp_mask = kp_mask.get()

    # Blend the keypoints.
    result = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    return result

##------------------------------------------------------------------------------
DATA_DIR='/home/rawalk/Desktop/foundational/mmpose/data/goliath/test_5000'
OUTPUT_DIR='/home/rawalk/Desktop/foundational/mmpose/Outputs/vis/misc/goliath_5000'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

IMAGES_DIR = os.path.join(DATA_DIR, 'images')
GT_DIR = os.path.join(DATA_DIR, 'keypoints')
PRED_DIR = os.path.join(DATA_DIR, 'pred_keypoints')

annotation_path = os.path.join(DATA_DIR, 'annotations', 'person_keypoints_test2023.json')
image_list_path = os.path.join(DATA_DIR, 'images.txt')

coco = COCO(annotation_path)

valid_image_ids = coco.getImgIds()
valid_image_infos = coco.loadImgs(valid_image_ids) ## valid coco annotations
valid_image_paths = []

for image_info in valid_image_infos:
    camera_id = image_info['camera_id']
    subject_id = image_info['subject_id']
    frame_index = image_info['frame_index']
    valid_image_paths.append(os.path.join(IMAGES_DIR, subject_id, camera_id, f'{frame_index}.jpg'))

## read image_list_path to a list
with open(image_list_path, "r") as f:
    lines = f.readlines()
    all_image_paths = [line.strip() for line in lines] ## sorted in order of hardness

image_paths = []

for image_path in all_image_paths:
    image_info = image_path.split('/')
    image_id = image_info[-1].replace('.jpg', '')
    camera_id = image_info[-2]
    subject_id = image_info[-3]

    image_path = os.path.join(IMAGES_DIR, subject_id, camera_id, f'{image_id}.jpg')

    if image_path in valid_image_paths:
        image_paths.append(image_path)

for idx, image_path in enumerate(image_paths):
    image_info = image_path.split('/')
    image_id = image_info[-1].replace('.jpg', '')
    camera_id = image_info[-2]
    subject_id = image_info[-3]

    image_path = os.path.join(IMAGES_DIR, subject_id, camera_id, f'{image_id}.jpg')
    gt_path = os.path.join(GT_DIR, subject_id, camera_id, f'{image_id}.npy')
    pred_path = os.path.join(PRED_DIR, subject_id, camera_id, f'{image_id}.npy')

    image = cv2.imread(image_path) ## BGR image

    gt_keypoints = np.load(gt_path)
    pred_keypoints = np.load(pred_path)

    print(f"Processing {idx + 1} out of {len(image_paths)}")

    gt_vis_image = goliath_vis_keypoints(image, gt_keypoints)
    pred_vis_image = goliath_vis_keypoints(image, pred_keypoints)

    vis_image = np.concatenate((gt_vis_image, pred_vis_image), axis=1)

    cv2.imwrite(f'{OUTPUT_DIR}/{str(idx).zfill(5)}.jpg', vis_image)
