import numpy as np
import os
import cv2
import json

data_dir = '/home/rawalk/Desktop/foundational/mmseg/data/coco'
save_dir = '/home/rawalk/Desktop/foundational/mmseg/data/coco/no_humans'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# set_name = 'val2017'
set_name = 'train2017'

annotation_dir = os.path.join(data_dir, 'annotations', 'person_keypoints_{}.json'.format(set_name))
image_dir = os.path.join(data_dir, set_name)

# Load the JSON annotations
with open(annotation_dir) as f:
    annotations = json.load(f)

# Extract image IDs with human annotations
human_image_ids = set()
for annotation in annotations['annotations']:
    if annotation.get('category_id') == 1:  # Assuming category_id 1 is for humans
        human_image_ids.add(annotation['image_id'])

# Filter out images without human annotations
no_human_images = []
for img in annotations['images']:
    if img['id'] not in human_image_ids:
        no_human_images.append(img['file_name'])

# Load these images
count = 0
for img_name in no_human_images:
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    if image is not None:
        count += 1
        print(f"{count}: Loaded {img_path}")

    save_img_path = os.path.join(save_dir, '{}_{}'.format(set_name, img_name))
    cv2.imwrite(save_img_path, image)

print(f"Loaded {count} images from {set_name} without human annotations.")
