import os
from tqdm import tqdm
import numpy as np
import cv2
import os
import concurrent.futures 

def process_image(image_name, base_dir, output_dir, offset):
    rgb_path = os.path.join(base_dir, 'images', image_name)
    gt_path = os.path.join(base_dir, 'gt_images', image_name)
    pred_path = os.path.join(base_dir, 'pred_images', image_name)
    save_path = os.path.join(output_dir, image_name)

    image = cv2.imread(rgb_path)
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)

    image = image[offset:, :, :]  # chop off the top

    target_height = gt.shape[0]
    assert gt.shape[0] == pred.shape[0]

    target_width = int(target_height * image.shape[1] / image.shape[0])
    image = cv2.resize(image, (target_width, target_height))

    vis_image = np.concatenate([image, gt, pred], axis=1)
    cv2.imwrite(save_path, vis_image)

def main():
    base_dir = '/mnt/home/rawalk/Desktop/sapiens/seg/data/iphone/point_clouds'
    image_dirs = ['images', 'gt_images', 'pred_images']
    output_dir = os.path.join(base_dir, 'combined')
    os.makedirs(output_dir, exist_ok=True)

    gt_images = set(os.listdir(os.path.join(base_dir, 'gt_images')))
    pred_images = set(os.listdir(os.path.join(base_dir, 'pred_images')))
    images = set(os.listdir(os.path.join(base_dir, 'images')))

    common_images = gt_images & pred_images & images
    common_images = sorted(common_images)

    offset = 600

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, image_name, base_dir, output_dir, offset) for image_name in common_images]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()  # Wait for each task to complete and raise any exceptions

    ## make a video called output.mp4 at fps 30. use ffmpeg
    command = "ffmpeg -framerate 30 -pattern_type glob -i '{}/*.png' -c:v mpeg4 -q:v 2 -pix_fmt yuv420p '{}/output.mp4'".format(output_dir, base_dir)
    os.system(command)

if __name__ == '__main__':
    main()
