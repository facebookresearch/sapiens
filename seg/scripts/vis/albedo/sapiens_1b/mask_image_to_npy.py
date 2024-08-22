import cv2
import numpy as np
import os

def process_masks(mask_dir):
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_seg_mask.png')]

    for mask_file in mask_files:
        print(mask_file)
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0]

        opacity = mask / 255.0
        mask = (opacity > 0.95) * 255

        npy_path = mask_path.replace('_seg_mask.png', '_image.npy')
        np.save(npy_path, mask)

if __name__ == '__main__':
    mask_dir = '/ablation_transient_a/junxuanli/MGR_test/masks'  # Change this to the path of your masks directory
    process_masks(mask_dir)
