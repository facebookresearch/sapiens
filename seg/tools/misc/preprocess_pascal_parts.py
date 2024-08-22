import os
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt

name_to_index = {
    'head': 1,
    'leye': 2,        # left eye
    'reye': 3,        # right eye
    'lear': 4,        # left ear
    'rear': 5,        # right ear
    'lebrow': 6,      # left eyebrow
    'rebrow': 7,      # right eyebrow
    'nose': 8,
    'mouth': 9,
    'hair': 10,
    'torso': 11,
    'neck': 12,
    'llarm': 13,      # left lower arm
    'luarm': 14,      # left upper arm
    'lhand': 15,      # left hand
    'rlarm': 16,      # right lower arm
    'ruarm': 17,      # right upper arm
    'rhand': 18,      # right hand
    'llleg': 19,      # left lower leg
    'luleg': 20,      # left upper leg
    'lfoot': 21,      # left foot
    'rlleg': 22,      # right lower leg
    'ruleg': 23,      # right upper leg
    'rfoot': 24       # right foot
}

colors = {
    1: [255, 0, 0],     # head - red
    2: [0, 255, 0],     # left eye - green
    3: [0, 0, 255],     # right eye - blue
    4: [255, 255, 0],   # left ear - yellow
    5: [255, 0, 255],   # right ear - magenta
    6: [0, 255, 255],   # left eyebrow - cyan
    7: [128, 0, 0],     # right eyebrow - dark red
    8: [0, 128, 0],     # nose - dark green
    9: [0, 0, 128],     # mouth - dark blue
    10: [128, 128, 0],  # hair - olive
    11: [128, 0, 128],  # torso - purple
    12: [0, 128, 128],  # neck - teal
    13: [255, 128, 0],  # left lower arm - orange
    14: [255, 0, 128],  # left upper arm - pink
    15: [128, 255, 0],  # left hand - lime
    16: [0, 128, 255],  # right lower arm - sky blue
    17: [128, 255, 255],# right upper arm - light cyan
    18: [255, 128, 128],# right hand - light red
    19: [128, 128, 255],# left lower leg - light blue
    20: [128, 255, 128],# left upper leg - light green
    21: [255, 255, 128],# left foot - light yellow
    22: [70, 130, 180], # right lower leg - steel blue
    23: [218, 165, 32], # right upper leg - golden rod
    24: [255, 69, 0]    # right foot - orange red
}

images_dir = '/home/rawalk/Desktop/foundational/mmseg/data/pascal/JPEGImages'
segmentations_dir = '/home/rawalk/Desktop/foundational/mmseg/data/pascal/Annotations_Part'
save_segmentations_dir = '/home/rawalk/Desktop/foundational/mmseg/data/pascal/Annotations_Human_Part'

# images_dir = '/Users/rawalk/Downloads/segmentation/pascal/JPEGImages'
# segmentations_dir = '/Users/rawalk/Downloads/segmentation/pascal/Annotations_Part'
# save_segmentations_dir = '/Users/rawalk/Downloads/segmentation/pascal/Annotations_Human_Part'

if not os.path.exists(save_segmentations_dir):
    os.mkdir(save_segmentations_dir)

segmentation_names = [name for name in sorted(os.listdir(segmentations_dir)) if name.endswith('.mat')] ## 10103 files

count = 0
for segmentation_name in segmentation_names:
    image_path = os.path.join(images_dir, segmentation_name.replace('.mat', '.jpg'))
    segmentation_path = os.path.join(segmentations_dir, segmentation_name)
    data = scipy.io.loadmat(segmentation_path)['anno'][0, 0]

    segmentation = None

    for obj in data['objects'][0, :]:
        class_name = obj['class'][0]
        if class_name != 'person':
            continue

        class_ind = obj['class_ind'][0, 0]
        n_parts = obj['parts'].shape[1]

        if n_parts == 0:
            continue

        for part in obj['parts'][0, :]:
            part_name = part['part_name'][0]
            part_index = name_to_index[part_name]
            mask = part['mask'] ## boolean mask

            if segmentation is None:
                segmentation = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

            segmentation[mask > 0] = part_index

    if segmentation is None:
        continue

    count += 1
    print('\033[92m{}: saving {}\033[0m'.format(count, segmentation_name))
    save_segmentation_path = os.path.join(save_segmentations_dir, segmentation_name.replace('.mat', '.png'))
    cv2.imwrite(save_segmentation_path, segmentation)

    # ## read it again
    # segmentation = cv2.imread(save_segmentation_path)

    # if len(segmentation.shape) == 3:
    #     segmentation = segmentation[:, :, 0]

    # Apply color palette to segmentation
    segmentation_color = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for part_index, color in colors.items():
        segmentation_color[segmentation == part_index] = color

    # Load original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Combine original image and colored segmentation
    combined_image = np.concatenate((original_image, segmentation_color), axis=1)
    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

    # Save the combined image using cv2
    save_path = os.path.join(save_segmentations_dir, segmentation_name.replace('.mat', '_vis.png'))
    cv2.imwrite(save_path, combined_image_bgr)
