from libcom import color_transfer
from libcom.utils.process_image import make_image_grid
import cv2
import numpy as np
import random
import os

idx = 37918

##------------------------------------------------------
image_name = '{}.jpg'.format(idx)
mask_name = '{}_seg.npy'.format(idx)

image = cv2.imread(image_name)
mask = np.load(mask_name)
## turn mask into binary mask
mask = (mask > 0).astype(np.uint8)

##------------------------------------------------------
def get_composite_image(image, mask, background_image):
    ## resize background image such that the height is equal to the height of the original image
    background_height = background_image.shape[0]
    background_width = background_image.shape[1]

    image_height = image.shape[0]
    image_width = image.shape[1]

    new_background_height = image_height
    new_background_width = int(new_background_height * background_width / background_height)

    background_image = cv2.resize(background_image, (new_background_width, new_background_height))

    # Crop the background image to the width of the original image
    if new_background_width > image_width:
        start_x = (new_background_width - image_width) // 2
        end_x = start_x + image_width
        background_image = background_image[:, start_x:end_x]

    ## make mask three channel
    mask_ = np.stack((mask,)*3, axis=-1)
    pasted_image = mask_*image + (1 - mask_)*background_image

    composite_mask = mask*255
    composite_image = color_transfer(pasted_image, composite_mask)

    grid_img  = make_image_grid([pasted_image, composite_mask, composite_image], cols=3)

    return pasted_image, composite_image, grid_img


##----------------------------------------------------
background_dir = '/home/rawalk/Desktop/foundational/mmseg/data/coco/no_humans'
background_images = sorted(os.listdir(background_dir))

num_samples = 50

for i in range(num_samples):

    print("Processing {}".format(i))

    ## randomly select a background image
    background_image_path = os.path.join(background_dir, random.choice(background_images))
    background_image = cv2.imread(background_image_path)

    pasted_image, composite_image, grid_img = get_composite_image(image, mask, background_image)

    # cv2.imwrite('pasted_image_{}.jpg'.format(i), pasted_image)
    # cv2.imwrite('composite_image_{}.jpg'.format(i), composite_image)
    cv2.imwrite('grid_{}.jpg'.format(i), grid_img)
