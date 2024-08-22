# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

## 34 classes in total
ORIGINAL_GOLIATH_CLASSES = (
    "Background",
    "Apparel",
    "Chair",
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Face_Neck",
    "Hair",
    "Headset",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Lower_Spandex",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Visible_Badge",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)

ORIGINAL_GOLIATH_PALETTE = [
    [50, 50, 50],
    [255, 218, 0],
    [102, 204, 0],
    [14, 0, 204],
    [0, 204, 160],
    [128, 200, 255],
    [255, 0, 109],
    [0, 255, 36],
    [189, 0, 204],
    [255, 0, 218],
    [0, 160, 204],
    [0, 255, 145],
    [204, 0, 131],
    [182, 0, 255],
    [255, 109, 0],
    [0, 255, 255],
    [72, 0, 255],
    [204, 43, 0],
    [204, 131, 0],
    [255, 0, 0],
    [72, 255, 0],
    [189, 204, 0],
    [182, 255, 0],
    [102, 0, 204],
    [32, 72, 204],
    [0, 145, 255],
    [14, 204, 0],
    [0, 128, 72],
    [204, 0, 43],
    [235, 205, 119],
    [115, 227, 112],
    [157, 113, 143],
    [132, 93, 50],
    [82, 21, 114],
]

## 6 classes to remove
REMOVE_CLASSES = (
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Visible_Badge",
    "Chair",
    "Lower_Spandex",
    "Headset",
)

## 34 - 6 = 28 classes left
GOLIATH_CLASSES = tuple(
    [x for x in ORIGINAL_GOLIATH_CLASSES if x not in REMOVE_CLASSES]
)
GOLIATH_PALETTE = [
    ORIGINAL_GOLIATH_PALETTE[idx]
    for idx in range(len(ORIGINAL_GOLIATH_CLASSES))
    if ORIGINAL_GOLIATH_CLASSES[idx] not in REMOVE_CLASSES
]

COCO_KPTS_COLORS = [
    [51, 153, 255],   # 0: nose
    [51, 153, 255],   # 1: left_eye
    [51, 153, 255],   # 2: right_eye
    [51, 153, 255],   # 3: left_ear
    [51, 153, 255],   # 4: right_ear
    [0, 255, 0],      # 5: left_shoulder
    [255, 128, 0],    # 6: right_shoulder
    [0, 255, 0],      # 7: left_elbow
    [255, 128, 0],    # 8: right_elbow
    [0, 255, 0],      # 9: left_wrist
    [255, 128, 0],    # 10: right_wrist
    [0, 255, 0],      # 11: left_hip
    [255, 128, 0],    # 12: right_hip
    [0, 255, 0],      # 13: left_knee
    [255, 128, 0],    # 14: right_knee
    [0, 255, 0],      # 15: left_ankle
    [255, 128, 0],    # 16: right_ankle
]

COCO_WHOLEBODY_KPTS_COLORS = [
    [51, 153, 255],   # 0: nose
    [51, 153, 255],   # 1: left_eye
    [51, 153, 255],   # 2: right_eye
    [51, 153, 255],   # 3: left_ear
    [51, 153, 255],   # 4: right_ear
    [0, 255, 0],      # 5: left_shoulder
    [255, 128, 0],    # 6: right_shoulder
    [0, 255, 0],      # 7: left_elbow
    [255, 128, 0],    # 8: right_elbow
    [0, 255, 0],      # 9: left_wrist
    [255, 128, 0],    # 10: right_wrist
    [0, 255, 0],      # 11: left_hip
    [255, 128, 0],    # 12: right_hip
    [0, 255, 0],      # 13: left_knee
    [255, 128, 0],    # 14: right_knee
    [0, 255, 0],      # 15: left_ankle
    [255, 128, 0],    # 16: right_ankle
    [255, 128, 0],    # 17: left_big_toe
    [255, 128, 0],    # 18: left_small_toe
    [255, 128, 0],    # 19: left_heel
    [255, 128, 0],    # 20: right_big_toe
    [255, 128, 0],    # 21: right_small_toe
    [255, 128, 0],    # 22: right_heel
    [255, 255, 255],  # 23: face-0
    [255, 255, 255],  # 24: face-1
    [255, 255, 255],  # 25: face-2
    [255, 255, 255],  # 26: face-3
    [255, 255, 255],  # 27: face-4
    [255, 255, 255],  # 28: face-5
    [255, 255, 255],  # 29: face-6
    [255, 255, 255],  # 30: face-7
    [255, 255, 255],  # 31: face-8
    [255, 255, 255],  # 32: face-9
    [255, 255, 255],  # 33: face-10
    [255, 255, 255],  # 34: face-11
    [255, 255, 255],  # 35: face-12
    [255, 255, 255],  # 36: face-13
    [255, 255, 255],  # 37: face-14
    [255, 255, 255],  # 38: face-15
    [255, 255, 255],  # 39: face-16
    [255, 255, 255],  # 40: face-17
    [255, 255, 255],  # 41: face-18
    [255, 255, 255],  # 42: face-19
    [255, 255, 255],  # 43: face-20
    [255, 255, 255],  # 44: face-21
    [255, 255, 255],  # 45: face-22
    [255, 255, 255],  # 46: face-23
    [255, 255, 255],  # 47: face-24
    [255, 255, 255],  # 48: face-25
    [255, 255, 255],  # 49: face-26
    [255, 255, 255],  # 50: face-27
    [255, 255, 255],  # 51: face-28
    [255, 255, 255],  # 52: face-29
    [255, 255, 255],  # 53: face-30
    [255, 255, 255],  # 54: face-31
    [255, 255, 255],  # 55: face-32
    [255, 255, 255],  # 56: face-33
    [255, 255, 255],  # 57: face-34
    [255, 255, 255],  # 58: face-35
    [255, 255, 255],  # 59: face-36
    [255, 255, 255],  # 60: face-37
    [255, 255, 255],  # 61: face-38
    [255, 255, 255],  # 62: face-39
    [255, 255, 255],  # 63: face-40
    [255, 255, 255],  # 64: face-41
    [255, 255, 255],  # 65: face-42
    [255, 255, 255],  # 66: face-43
    [255, 255, 255],  # 67: face-44
    [255, 255, 255],  # 68: face-45
    [255, 255, 255],  # 69: face-46
    [255, 255, 255],  # 70: face-47
    [255, 255, 255],  # 71: face-48
    [255, 255, 255],  # 72: face-49
    [255, 255, 255],  # 73: face-50
    [255, 255, 255],  # 74: face-51
    [255, 255, 255],  # 75: face-52
    [255, 255, 255],  # 76: face-53
    [255, 255, 255],  # 77: face-54
    [255, 255, 255],  # 78: face-55
    [255, 255, 255],  # 79: face-56
    [255, 255, 255],  # 80: face-57
    [255, 255, 255],  # 81: face-58
    [255, 255, 255],  # 82: face-59
    [255, 255, 255],  # 83: face-60
    [255, 255, 255],  # 84: face-61
    [255, 255, 255],  # 85: face-62
    [255, 255, 255],  # 86: face-63
    [255, 255, 255],  # 87: face-64
    [255, 255, 255],  # 88: face-65
    [255, 255, 255],  # 89: face-66
    [255, 255, 255],  # 90: face-67
    [255, 255, 255],  # 91: left_hand_root
    [255, 128, 0],    # 92: left_thumb1
    [255, 128, 0],    # 93: left_thumb2
    [255, 128, 0],    # 94: left_thumb3
    [255, 128, 0],    # 95: left_thumb4
    [255, 153, 255],  # 96: left_forefinger1
    [255, 153, 255],  # 97: left_forefinger2
    [255, 153, 255],  # 98: left_forefinger3
    [255, 153, 255],  # 99: left_forefinger4
    [102, 178, 255],  # 100: left_middle_finger1
    [102, 178, 255],  # 101: left_middle_finger2
    [102, 178, 255],  # 102: left_middle_finger3
    [102, 178, 255],  # 103: left_middle_finger4
    [255, 51, 51],    # 104: left_ring_finger1
    [255, 51, 51],    # 105: left_ring_finger2
    [255, 51, 51],    # 106: left_ring_finger3
    [255, 51, 51],    # 107: left_ring_finger4
    [0, 255, 0],      # 108: left_pinky_finger1
    [0, 255, 0],      # 109: left_pinky_finger2
    [0, 255, 0],      # 110: left_pinky_finger3
    [0, 255, 0],      # 111: left_pinky_finger4
    [255, 255, 255],  # 112: right_hand_root
    [255, 128, 0],    # 113: right_thumb1
    [255, 128, 0],    # 114: right_thumb2
    [255, 128, 0],    # 115: right_thumb3
    [255, 128, 0],    # 116: right_thumb4
    [255, 153, 255],  # 117: right_forefinger1
    [255, 153, 255],  # 118: right_forefinger2
    [255, 153, 255],  # 119: right_forefinger3
    [255, 153, 255],  # 120: right_forefinger4
    [102, 178, 255],  # 121: right_middle_finger1
    [102, 178, 255],  # 122: right_middle_finger2
    [102, 178, 255],  # 123: right_middle_finger3
    [102, 178, 255],  # 124: right_middle_finger4
    [255, 51, 51],    # 125: right_ring_finger1
    [255, 51, 51],    # 126: right_ring_finger2
    [255, 51, 51],    # 127: right_ring_finger3
    [255, 51, 51],    # 128: right_ring_finger4
    [0, 255, 0],      # 129: right_pinky_finger1
    [0, 255, 0],      # 130: right_pinky_finger2
    [0, 255, 0],      # 131: right_pinky_finger3
    [0, 255, 0],      # 132: right_pinky_finger4
]


GOLIATH_KPTS_COLORS = [
    [51, 153, 255],   # 0: nose
    [51, 153, 255],   # 1: left_eye
    [51, 153, 255],   # 2: right_eye
    [51, 153, 255],   # 3: left_ear
    [51, 153, 255],   # 4: right_ear
    [51, 153, 255],   # 5: left_shoulder
    [51, 153, 255],   # 6: right_shoulder
    [51, 153, 255],   # 7: left_elbow
    [51, 153, 255],   # 8: right_elbow
    [51, 153, 255],   # 9: left_hip
    [51, 153, 255],   # 10: right_hip
    [51, 153, 255],   # 11: left_knee
    [51, 153, 255],   # 12: right_knee
    [51, 153, 255],   # 13: left_ankle
    [51, 153, 255],   # 14: right_ankle
    [51, 153, 255],   # 15: left_big_toe
    [51, 153, 255],   # 16: left_small_toe
    [51, 153, 255],   # 17: left_heel
    [51, 153, 255],   # 18: right_big_toe
    [51, 153, 255],   # 19: right_small_toe
    [51, 153, 255],   # 20: right_heel
    [51, 153, 255],   # 21: right_thumb4
    [51, 153, 255],   # 22: right_thumb3
    [51, 153, 255],   # 23: right_thumb2
    [51, 153, 255],   # 24: right_thumb_third_joint
    [51, 153, 255],   # 25: right_forefinger4
    [51, 153, 255],   # 26: right_forefinger3
    [51, 153, 255],   # 27: right_forefinger2
    [51, 153, 255],   # 28: right_forefinger_third_joint
    [51, 153, 255],   # 29: right_middle_finger4
    [51, 153, 255],   # 30: right_middle_finger3
    [51, 153, 255],   # 31: right_middle_finger2
    [51, 153, 255],   # 32: right_middle_finger_third_joint
    [51, 153, 255],   # 33: right_ring_finger4
    [51, 153, 255],   # 34: right_ring_finger3
    [51, 153, 255],   # 35: right_ring_finger2
    [51, 153, 255],   # 36: right_ring_finger_third_joint
    [51, 153, 255],   # 37: right_pinky_finger4
    [51, 153, 255],   # 38: right_pinky_finger3
    [51, 153, 255],   # 39: right_pinky_finger2
    [51, 153, 255],   # 40: right_pinky_finger_third_joint
    [51, 153, 255],   # 41: right_wrist
    [51, 153, 255],   # 42: left_thumb4
    [51, 153, 255],   # 43: left_thumb3
    [51, 153, 255],   # 44: left_thumb2
    [51, 153, 255],   # 45: left_thumb_third_joint
    [51, 153, 255],   # 46: left_forefinger4
    [51, 153, 255],   # 47: left_forefinger3
    [51, 153, 255],   # 48: left_forefinger2
    [51, 153, 255],   # 49: left_forefinger_third_joint
    [51, 153, 255],   # 50: left_middle_finger4
    [51, 153, 255],   # 51: left_middle_finger3
    [51, 153, 255],   # 52: left_middle_finger2
    [51, 153, 255],   # 53: left_middle_finger_third_joint
    [51, 153, 255],   # 54: left_ring_finger4
    [51, 153, 255],   # 55: left_ring_finger3
    [51, 153, 255],   # 56: left_ring_finger2
    [51, 153, 255],   # 57: left_ring_finger_third_joint
    [51, 153, 255],   # 58: left_pinky_finger4
    [51, 153, 255],   # 59: left_pinky_finger3
    [51, 153, 255],   # 60: left_pinky_finger2
    [51, 153, 255],   # 61: left_pinky_finger_third_joint
    [51, 153, 255],   # 62: left_wrist
    [51, 153, 255],   # 63: left_olecranon
    [51, 153, 255],   # 64: right_olecranon
    [51, 153, 255],   # 65: left_cubital_fossa
    [51, 153, 255],   # 66: right_cubital_fossa
    [51, 153, 255],   # 67: left_acromion
    [51, 153, 255],   # 68: right_acromion
    [51, 153, 255],   # 69: neck
    [255, 255, 255],  # 70: center_of_glabella
    [255, 255, 255],  # 71: center_of_nose_root
    [255, 255, 255],  # 72: tip_of_nose_bridge
    [255, 255, 255],  # 73: midpoint_1_of_nose_bridge
    [255, 255, 255],  # 74: midpoint_2_of_nose_bridge
    [255, 255, 255],  # 75: midpoint_3_of_nose_bridge
    [255, 255, 255],  # 76: center_of_labiomental_groove
    [255, 255, 255],  # 77: tip_of_chin
    [255, 255, 255],  # 78: upper_startpoint_of_r_eyebrow
    [255, 255, 255],  # 79: lower_startpoint_of_r_eyebrow
    [255, 255, 255],  # 80: end_of_r_eyebrow
    [255, 255, 255],  # 81: upper_midpoint_1_of_r_eyebrow
    [255, 255, 255],  # 82: lower_midpoint_1_of_r_eyebrow
    [255, 255, 255],  # 83: upper_midpoint_2_of_r_eyebrow
    [255, 255, 255],  # 84: upper_midpoint_3_of_r_eyebrow
    [255, 255, 255],  # 85: lower_midpoint_2_of_r_eyebrow
    [255, 255, 255],  # 86: lower_midpoint_3_of_r_eyebrow
    [255, 255, 255],  # 87: upper_startpoint_of_l_eyebrow
    [255, 255, 255],  # 88: lower_startpoint_of_l_eyebrow
    [255, 255, 255],  # 89: end_of_l_eyebrow
    [255, 255, 255],  # 90: upper_midpoint_1_of_l_eyebrow
    [255, 255, 255],  # 91: lower_midpoint_1_of_l_eyebrow
    [255, 255, 255],  # 92: upper_midpoint_2_of_l_eyebrow
    [255, 255, 255],  # 93: upper_midpoint_3_of_l_eyebrow
    [255, 255, 255],  # 94: lower_midpoint_2_of_l_eyebrow
    [255, 255, 255],  # 95: lower_midpoint_3_of_l_eyebrow
    [192, 64, 128],   # 96: l_inner_end_of_upper_lash_line
    [192, 64, 128],   # 97: l_outer_end_of_upper_lash_line
    [192, 64, 128],   # 98: l_centerpoint_of_upper_lash_line
    [192, 64, 128],   # 99: l_midpoint_2_of_upper_lash_line
    [192, 64, 128],   # 100: l_midpoint_1_of_upper_lash_line
    [192, 64, 128],   # 101: l_midpoint_6_of_upper_lash_line
    [192, 64, 128],   # 102: l_midpoint_5_of_upper_lash_line
    [192, 64, 128],   # 103: l_midpoint_4_of_upper_lash_line
    [192, 64, 128],   # 104: l_midpoint_3_of_upper_lash_line
    [192, 64, 128],   # 105: l_outer_end_of_upper_eyelid_line
    [192, 64, 128],   # 106: l_midpoint_6_of_upper_eyelid_line
    [192, 64, 128],   # 107: l_midpoint_2_of_upper_eyelid_line
    [192, 64, 128],   # 108: l_midpoint_5_of_upper_eyelid_line
    [192, 64, 128],   # 109: l_centerpoint_of_upper_eyelid_line
    [192, 64, 128],   # 110: l_midpoint_4_of_upper_eyelid_line
    [192, 64, 128],   # 111: l_midpoint_1_of_upper_eyelid_line
    [192, 64, 128],   # 112: l_midpoint_3_of_upper_eyelid_line
    [192, 64, 128],   # 113: l_midpoint_6_of_upper_crease_line
    [192, 64, 128],   # 114: l_midpoint_2_of_upper_crease_line
    [192, 64, 128],   # 115: l_midpoint_5_of_upper_crease_line
    [192, 64, 128],   # 116: l_centerpoint_of_upper_crease_line
    [192, 64, 128],   # 117: l_midpoint_4_of_upper_crease_line
    [192, 64, 128],   # 118: l_midpoint_1_of_upper_crease_line
    [192, 64, 128],   # 119: l_midpoint_3_of_upper_crease_line
    [64, 32, 192],    # 120: r_inner_end_of_upper_lash_line
    [64, 32, 192],    # 121: r_outer_end_of_upper_lash_line
    [64, 32, 192],    # 122: r_centerpoint_of_upper_lash_line
    [64, 32, 192],    # 123: r_midpoint_1_of_upper_lash_line
    [64, 32, 192],    # 124: r_midpoint_2_of_upper_lash_line
    [64, 32, 192],    # 125: r_midpoint_3_of_upper_lash_line
    [64, 32, 192],    # 126: r_midpoint_4_of_upper_lash_line
    [64, 32, 192],    # 127: r_midpoint_5_of_upper_lash_line
    [64, 32, 192],    # 128: r_midpoint_6_of_upper_lash_line
    [64, 32, 192],    # 129: r_outer_end_of_upper_eyelid_line
    [64, 32, 192],    # 130: r_midpoint_3_of_upper_eyelid_line
    [64, 32, 192],    # 131: r_midpoint_1_of_upper_eyelid_line
    [64, 32, 192],    # 132: r_midpoint_4_of_upper_eyelid_line
    [64, 32, 192],    # 133: r_centerpoint_of_upper_eyelid_line
    [64, 32, 192],    # 134: r_midpoint_5_of_upper_eyelid_line
    [64, 32, 192],    # 135: r_midpoint_2_of_upper_eyelid_line
    [64, 32, 192],    # 136: r_midpoint_6_of_upper_eyelid_line
    [64, 32, 192],    # 137: r_midpoint_3_of_upper_crease_line
    [64, 32, 192],    # 138: r_midpoint_1_of_upper_crease_line
    [64, 32, 192],    # 139: r_midpoint_4_of_upper_crease_line
    [64, 32, 192],    # 140: r_centerpoint_of_upper_crease_line
    [64, 32, 192],    # 141: r_midpoint_5_of_upper_crease_line
    [64, 32, 192],    # 142: r_midpoint_2_of_upper_crease_line
    [64, 32, 192],    # 143: r_midpoint_6_of_upper_crease_line
    [64, 192, 128],   # 144: l_inner_end_of_lower_lash_line
    [64, 192, 128],   # 145: l_outer_end_of_lower_lash_line
    [64, 192, 128],   # 146: l_centerpoint_of_lower_lash_line
    [64, 192, 128],   # 147: l_midpoint_2_of_lower_lash_line
    [64, 192, 128],   # 148: l_midpoint_1_of_lower_lash_line
    [64, 192, 128],   # 149: l_midpoint_6_of_lower_lash_line
    [64, 192, 128],   # 150: l_midpoint_5_of_lower_lash_line
    [64, 192, 128],   # 151: l_midpoint_4_of_lower_lash_line
    [64, 192, 128],   # 152: l_midpoint_3_of_lower_lash_line
    [64, 192, 128],   # 153: l_outer_end_of_lower_eyelid_line
    [64, 192, 128],   # 154: l_midpoint_6_of_lower_eyelid_line
    [64, 192, 128],   # 155: l_midpoint_2_of_lower_eyelid_line
    [64, 192, 128],   # 156: l_midpoint_5_of_lower_eyelid_line
    [64, 192, 128],   # 157: l_centerpoint_of_lower_eyelid_line
    [64, 192, 128],   # 158: l_midpoint_4_of_lower_eyelid_line
    [64, 192, 128],   # 159: l_midpoint_1_of_lower_eyelid_line
    [64, 192, 128],   # 160: l_midpoint_3_of_lower_eyelid_line
    [64, 192, 32],    # 161: r_inner_end_of_lower_lash_line
    [64, 192, 32],    # 162: r_outer_end_of_lower_lash_line
    [64, 192, 32],    # 163: r_centerpoint_of_lower_lash_line
    [64, 192, 32],    # 164: r_midpoint_1_of_lower_lash_line
    [64, 192, 32],    # 165: r_midpoint_2_of_lower_lash_line
    [64, 192, 32],    # 166: r_midpoint_3_of_lower_lash_line
    [64, 192, 32],    # 167: r_midpoint_4_of_lower_lash_line
    [64, 192, 32],    # 168: r_midpoint_5_of_lower_lash_line
    [64, 192, 32],    # 169: r_midpoint_6_of_lower_lash_line
    [64, 192, 32],    # 170: r_outer_end_of_lower_eyelid_line
    [64, 192, 32],    # 171: r_midpoint_3_of_lower_eyelid_line
    [64, 192, 32],    # 172: r_midpoint_1_of_lower_eyelid_line
    [64, 192, 32],    # 173: r_midpoint_4_of_lower_eyelid_line
    [64, 192, 32],    # 174: r_centerpoint_of_lower_eyelid_line
    [64, 192, 32],    # 175: r_midpoint_5_of_lower_eyelid_line
    [64, 192, 32],    # 176: r_midpoint_2_of_lower_eyelid_line
    [64, 192, 32],    # 177: r_midpoint_6_of_lower_eyelid_line
    [0, 192, 0],      # 178: tip_of_nose
    [0, 192, 0],      # 179: bottom_center_of_nose
    [0, 192, 0],      # 180: r_outer_corner_of_nose
    [0, 192, 0],      # 181: l_outer_corner_of_nose
    [0, 192, 0],      # 182: inner_corner_of_r_nostril
    [0, 192, 0],      # 183: outer_corner_of_r_nostril
    [0, 192, 0],      # 184: upper_corner_of_r_nostril
    [0, 192, 0],      # 185: inner_corner_of_l_nostril
    [0, 192, 0],      # 186: outer_corner_of_l_nostril
    [0, 192, 0],      # 187: upper_corner_of_l_nostril
    [192, 0, 0],      # 188: r_outer_corner_of_mouth
    [192, 0, 0],      # 189: l_outer_corner_of_mouth
    [192, 0, 0],      # 190: center_of_cupid_bow
    [192, 0, 0],      # 191: center_of_lower_outer_lip
    [192, 0, 0],      # 192: midpoint_1_of_upper_outer_lip
    [192, 0, 0],      # 193: midpoint_2_of_upper_outer_lip
    [192, 0, 0],      # 194: midpoint_1_of_lower_outer_lip
    [192, 0, 0],      # 195: midpoint_2_of_lower_outer_lip
    [192, 0, 0],      # 196: midpoint_3_of_upper_outer_lip
    [192, 0, 0],      # 197: midpoint_4_of_upper_outer_lip
    [192, 0, 0],      # 198: midpoint_5_of_upper_outer_lip
    [192, 0, 0],      # 199: midpoint_6_of_upper_outer_lip
    [192, 0, 0],      # 200: midpoint_3_of_lower_outer_lip
    [192, 0, 0],      # 201: midpoint_4_of_lower_outer_lip
    [192, 0, 0],      # 202: midpoint_5_of_lower_outer_lip
    [192, 0, 0],      # 203: midpoint_6_of_lower_outer_lip
    [0, 192, 192],    # 204: r_inner_corner_of_mouth
    [0, 192, 192],    # 205: l_inner_corner_of_mouth
    [0, 192, 192],    # 206: center_of_upper_inner_lip
    [0, 192, 192],    # 207: center_of_lower_inner_lip
    [0, 192, 192],    # 208: midpoint_1_of_upper_inner_lip
    [0, 192, 192],    # 209: midpoint_2_of_upper_inner_lip
    [0, 192, 192],    # 210: midpoint_1_of_lower_inner_lip
    [0, 192, 192],    # 211: midpoint_2_of_lower_inner_lip
    [0, 192, 192],    # 212: midpoint_3_of_upper_inner_lip
    [0, 192, 192],    # 213: midpoint_4_of_upper_inner_lip
    [0, 192, 192],    # 214: midpoint_5_of_upper_inner_lip
    [0, 192, 192],    # 215: midpoint_6_of_upper_inner_lip
    [0, 192, 192],    # 216: midpoint_3_of_lower_inner_lip
    [0, 192, 192],    # 217: midpoint_4_of_lower_inner_lip
    [0, 192, 192],    # 218: midpoint_5_of_lower_inner_lip
    [0, 192, 192],    # 219: midpoint_6_of_lower_inner_lip. teeths removed
    [200, 200, 0],    # 256: l_top_end_of_inferior_crus
    [200, 200, 0],    # 257: l_top_end_of_superior_crus
    [200, 200, 0],    # 258: l_start_of_antihelix
    [200, 200, 0],    # 259: l_end_of_antihelix
    [200, 200, 0],    # 260: l_midpoint_1_of_antihelix
    [200, 200, 0],    # 261: l_midpoint_1_of_inferior_crus
    [200, 200, 0],    # 262: l_midpoint_2_of_antihelix
    [200, 200, 0],    # 263: l_midpoint_3_of_antihelix
    [200, 200, 0],    # 264: l_point_1_of_inner_helix
    [200, 200, 0],    # 265: l_point_2_of_inner_helix
    [200, 200, 0],    # 266: l_point_3_of_inner_helix
    [200, 200, 0],    # 267: l_point_4_of_inner_helix
    [200, 200, 0],    # 268: l_point_5_of_inner_helix
    [200, 200, 0],    # 269: l_point_6_of_inner_helix
    [200, 200, 0],    # 270: l_point_7_of_inner_helix
    [200, 200, 0],    # 271: l_highest_point_of_antitragus
    [200, 200, 0],    # 272: l_bottom_point_of_tragus
    [200, 200, 0],    # 273: l_protruding_point_of_tragus
    [200, 200, 0],    # 274: l_top_point_of_tragus
    [200, 200, 0],    # 275: l_start_point_of_crus_of_helix
    [200, 200, 0],    # 276: l_deepest_point_of_concha
    [200, 200, 0],    # 277: l_tip_of_ear_lobe
    [200, 200, 0],    # 278: l_midpoint_between_22_15
    [200, 200, 0],    # 279: l_bottom_connecting_point_of_ear_lobe
    [200, 200, 0],    # 280: l_top_connecting_point_of_helix
    [200, 200, 0],    # 281: l_point_8_of_inner_helix
    [0, 200, 200],    # 282: r_top_end_of_inferior_crus
    [0, 200, 200],    # 283: r_top_end_of_superior_crus
    [0, 200, 200],    # 284: r_start_of_antihelix
    [0, 200, 200],    # 285: r_end_of_antihelix
    [0, 200, 200],    # 286: r_midpoint_1_of_antihelix
    [0, 200, 200],    # 287: r_midpoint_1_of_inferior_crus
    [0, 200, 200],    # 288: r_midpoint_2_of_antihelix
    [0, 200, 200],    # 289: r_midpoint_3_of_antihelix
    [0, 200, 200],    # 290: r_point_1_of_inner_helix
    [0, 200, 200],    # 291: r_point_8_of_inner_helix
    [0, 200, 200],    # 292: r_point_3_of_inner_helix
    [0, 200, 200],    # 293: r_point_4_of_inner_helix
    [0, 200, 200],    # 294: r_point_5_of_inner_helix
    [0, 200, 200],    # 295: r_point_6_of_inner_helix
    [0, 200, 200],    # 296: r_point_7_of_inner_helix
    [0, 200, 200],    # 297: r_highest_point_of_antitragus
    [0, 200, 200],    # 298: r_bottom_point_of_tragus
    [0, 200, 200],    # 299: r_protruding_point_of_tragus
    [0, 200, 200],    # 300: r_top_point_of_tragus
    [0, 200, 200],    # 301: r_start_point_of_crus_of_helix
    [0, 200, 200],    # 302: r_deepest_point_of_concha
    [0, 200, 200],    # 303: r_tip_of_ear_lobe
    [0, 200, 200],    # 304: r_midpoint_between_22_15
    [0, 200, 200],    # 305: r_bottom_connecting_point_of_ear_lobe
    [0, 200, 200],    # 306: r_top_connecting_point_of_helix
    [0, 200, 200],    # 307: r_point_2_of_inner_helix
    [128, 192, 64],   # 308: l_center_of_iris
    [128, 192, 64],   # 309: l_border_of_iris_3
    [128, 192, 64],   # 310: l_border_of_iris_midpoint_1
    [128, 192, 64],   # 311: l_border_of_iris_12
    [128, 192, 64],   # 312: l_border_of_iris_midpoint_4
    [128, 192, 64],   # 313: l_border_of_iris_9
    [128, 192, 64],   # 314: l_border_of_iris_midpoint_3
    [128, 192, 64],   # 315: l_border_of_iris_6
    [128, 192, 64],   # 316: l_border_of_iris_midpoint_2
    [192, 32, 64],    # 317: r_center_of_iris
    [192, 32, 64],    # 318: r_border_of_iris_3
    [192, 32, 64],    # 319: r_border_of_iris_midpoint_1
    [192, 32, 64],    # 320: r_border_of_iris_12
    [192, 32, 64],    # 321: r_border_of_iris_midpoint_4
    [192, 32, 64],    # 322: r_border_of_iris_9
    [192, 32, 64],    # 323: r_border_of_iris_midpoint_3
    [192, 32, 64],    # 324: r_border_of_iris_6
    [192, 32, 64],    # 325: r_border_of_iris_midpoint_2
    [192, 128, 64],   # 326: l_center_of_pupil
    [192, 128, 64],   # 327: l_border_of_pupil_3
    [192, 128, 64],   # 328: l_border_of_pupil_midpoint_1
    [192, 128, 64],   # 329: l_border_of_pupil_12
    [192, 128, 64],   # 330: l_border_of_pupil_midpoint_4
    [192, 128, 64],   # 331: l_border_of_pupil_9
    [192, 128, 64],   # 332: l_border_of_pupil_midpoint_3
    [192, 128, 64],   # 333: l_border_of_pupil_6
    [192, 128, 64],   # 334: l_border_of_pupil_midpoint_2
    [32, 192, 192],   # 335: r_center_of_pupil
    [32, 192, 192],   # 336: r_border_of_pupil_3
    [32, 192, 192],   # 337: r_border_of_pupil_midpoint_1
    [32, 192, 192],   # 338: r_border_of_pupil_12
    [32, 192, 192],   # 339: r_border_of_pupil_midpoint_4
    [32, 192, 192],   # 340: r_border_of_pupil_9
    [32, 192, 192],   # 341: r_border_of_pupil_midpoint_3
    [32, 192, 192],   # 342: r_border_of_pupil_6
    [32, 192, 192],   # 343: r_border_of_pupil_midpoint_2
]
