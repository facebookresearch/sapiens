# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import cv2

def create_video_from_images(folder_path):
    # Grab all the image files from the given directory
    image_files = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    image_files.sort()  # Assuming alphanumeric ordering is desired

    # Ensure there are image files in the directory
    if not image_files:
        print("No images found in the specified directory!")
        return
    
    # Grab the dimensions from the first image
    frame = cv2.imread(os.path.join(folder_path, image_files[0]))
    h, w, layers = frame.shape
    size = (w, h)

    # Define the codec and create VideoWriter object
    fps = 30
    out = cv2.VideoWriter(os.path.join(folder_path, "output.mp4"), cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    
    for i in range(len(image_files)):
        img_path = os.path.join(folder_path, image_files[i])
        img = cv2.imread(img_path)
        out.write(img)
        if i % 50 == 0:
            print(f"Processed {i} images.")

    out.release()
    print(f"Video saved to: {os.path.join(folder_path, 'output.mp4')}")

if __name__ == "__main__":
    # folder_path = sys.argv[1]
    # folder_path = '/home/rawalk/Desktop/foundational/mmpose/Outputs/goliath/goliath/s--20190524--1430--4911137--GHS/400130'
    # folder_path = '/home/rawalk/Desktop/foundational/mmpose/Outputs/goliath/goliath/s--20190524--1430--4911137--GHS/400156'
    # folder_path = '/home/rawalk/Desktop/foundational/mmpose/Outputs/goliath/goliath/s--20190524--1430--4911137--GHS/400190'
    folder_path = '/home/rawalk/Downloads/output6/output/internal_hand1'

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        create_video_from_images(folder_path)
    else:
        print(f"Error: The provided path '{folder_path}' is not a valid directory.")

