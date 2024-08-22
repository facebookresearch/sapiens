# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

IMAGE_DIR='/home/rawalk/Desktop/sapiens/pose/demo/data/itw_videos/reel1'
FEATURE_DIR='/home/rawalk/Desktop/sapiens/pretrain/Outputs/vis/itw_videos/reel1'
OUTPUT_DIR = os.path.join(FEATURE_DIR, 'pca')

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

## load all .npy files
feature_names = sorted([x for x in os.listdir(FEATURE_DIR) if x.endswith('.npy')])
image_names = sorted([x for x in os.listdir(IMAGE_DIR) if x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')])
num_images = len(feature_names)

features = []
for i, feature_name in tqdm(enumerate(feature_names), total=num_images):
    feature_path = os.path.join(FEATURE_DIR, feature_name)
    feature = np.load(feature_path) ## embed_dim x H x W. For sapien_1b: 1536 x 64 x 64
    embed_dim, h, w = feature.shape

    feature = feature.reshape(embed_dim, -1) ## flatten, 1536 x (64*64)
    feature = feature.T ## (64*64) x 1536
    feature = feature.reshape(1, -1, embed_dim) ## 1 x (64*64) x 1536
    features.append(feature)

features = np.concatenate(features, axis=0) ## N x (64*64) x 1536
features = features.reshape(-1, embed_dim) ## (N x (64*64)) x 1536

## PCA
n_components = 3
num_vis_images = min(10, num_images)

pca = PCA(n_components=n_components)
pca.fit(features)
pca_features = pca.transform(features) ## (N x (64*64)) x 3


# Dynamic thresholding for background vs foreground
background_percentile = 20  # Percentage of pixels to be considered as background
threshold = np.percentile(pca_features[:, 0], background_percentile)

background = pca_features[:, 0] <= threshold
foreground = ~background

## again fit PCA on foreground only
pca.fit(pca_features[foreground])
features_foreground = pca.transform(pca_features[foreground])

for i in range(n_components):
    features_foreground[:, i] = (features_foreground[:, i] - features_foreground[:, i].min()) \
                                / (features_foreground[:, i].max() - features_foreground[:, i].min())

rgb = pca_features.copy()
rgb[background] = 0
rgb[foreground] = features_foreground

rgb = rgb.reshape(num_images, h, w, n_components)

for i in range(num_vis_images):
    rgb_image = cv2.imread(os.path.join(IMAGE_DIR, image_names[i]))
    image_height, image_width, _ = rgb_image.shape

    feature_image = rgb[i]

    ## resize by to rgb image size
    feature_image = cv2.resize(feature_image, (image_width, image_height))

    ## convert to bgr
    feature_image = feature_image[:, :, ::-1]
    feature_image = (feature_image * 255).astype('uint8')

    vis_image = np.concatenate([rgb_image, feature_image], axis=1)

    save_path = os.path.join(OUTPUT_DIR, image_names[i].replace('.jpg', '_pca.png'))
    cv2.imwrite(save_path, vis_image)

####-------------------------------------------------------------------------
# num_vis_images = min(10, num_images)
# for component_id in range(n_components):
#     print("Component {}".format(component_id))
#     for i in range(num_vis_images):
#         ## read rgb image
#         rgb_image = cv2.imread(os.path.join(IMAGE_DIR, image_names[i]))
#         image_height, image_width, _ = rgb_image.shape

#         img = pca_features[i * h * w : (i+1) * h * w, component_id].reshape(h, w)

#         ## normalize the values between 0 and 1
#         img = (img - np.min(img)) / (np.max(img) - np.min(img))

#         ## resize by to rgb image size
#         img = cv2.resize(img, (image_width, image_height))

#         ## visualize as heatmap inferno
#         img = cv2.applyColorMap((img * 255).astype('uint8'), cv2.COLORMAP_INFERNO)

#         ## save the image
#         save_path = os.path.join(OUTPUT_DIR, feature_names[i].replace('.npy', '_pca_{}.png'.format(component_id)))

#         vis_image = np.concatenate([rgb_image, img], axis=1)

#         cv2.imwrite(save_path, vis_image)
