import os
import time
from argparse import ArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import nopdb
from PIL import Image
import mmengine
import mmcv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTModel
from transformers import AutoImageProcessor, AutoModel
import timm

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    assert args.output_root != ''
    assert args.input != ''

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))

    ##--------------------at devserver------------------------
    # model = timm.create_model('vit_giant_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0)
    # model = model.eval()
    # torch.save(model.state_dict(), '/home/rawalk/Downloads/vit_giant_patch14_reg4_dinov2_lvd142m.pth')
    ## copy to dgx

    model = timm.create_model('vit_giant_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=0)  # remove classifier nn.Linear
    model.load_state_dict(torch.load('/mnt/home/rawalk/drive/pretrain/checkpoints/registers-dinov2-giant/vit_giant_patch14_reg4_dinov2_lvd142m.pth'))
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    for i, image_name in tqdm(enumerate(image_names), total=len(image_names)):
        image_path = os.path.join(input_dir, image_name)  # Join the directory path with the image file name

        image = Image.open(image_path)

        with nopdb.capture_call(model.blocks[-1].attn.forward) as attn_call:
            output = model(transforms(image).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

        query, key, value = attn_call.locals['q'], attn_call.locals['k'], attn_call.locals['v']
        scale = query.size(-1)**0.5
        attn_weight_raw = query @ key.transpose(-2, -1) / scale
        attn_weight_raw = torch.softmax(attn_weight_raw, dim=-1) ## [batch_size, num_heads, seq_length, seq_length], 1 x 24 x 4097 x 4097

        num_heads = attn_weight_raw.size(1) ## 24
        patch_size = 14
        w_featmap = 518 // patch_size
        h_featmap = 518 // patch_size

        # Extract the attention scores for the first token (originally the class token) with respect to all patch tokens
        # Since the class token has been removed, the first token is now the first patch token
        for vis_token_index in range(w_featmap * h_featmap):
            print('image:{} vis_token_index:{}/{}'.format(image_name, vis_token_index, w_featmap*h_featmap))
            attn_weight = attn_weight_raw[0, :, vis_token_index, 5:].reshape(num_heads, -1) ## 24 x 1369 - remove the cls and 4 register tokens

            # Reshape and interpolate the original attention weights
            attn_weight = attn_weight.reshape(num_heads, w_featmap, h_featmap) ## 24 x 16 x 16

            attn_weight = F.interpolate(attn_weight.unsqueeze(0), scale_factor=8, mode="nearest")[0].detach().cpu().numpy() ## 256 x 256

            # Visualize and save
            output_dir = os.path.join(args.output_root, os.path.basename(image_path).replace('.jpg', '_attn').replace('.jpeg', '_attn').replace('.png', '_attn'))
            os.makedirs(output_dir, exist_ok=True)

            grid_size = 5  # Grid size for the collage (5x5)
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

            for i in range(grid_size):
                for j in range(grid_size):
                    ax = axes[i, j]
                    idx = i * grid_size + j  # Index for the current subplot
                    if idx < num_heads:
                        ax.imshow(attn_weight[idx])
                    ax.axis('off')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(os.path.join(output_dir, '{}.png'.format(vis_token_index)))
            plt.close()

            ## also save all heads as there own image
            vis_token_index_dir = os.path.join(output_dir, str(vis_token_index))
            os.makedirs(vis_token_index_dir, exist_ok=True)

            for head_idx in range(num_heads):
                head_attn = attn_weight[head_idx]
                # Normalize the attention weights to the range [0, 255] for visualization
                head_attn_normalized = cv2.normalize(head_attn, None, 0, 255, cv2.NORM_MINMAX)
                # Apply the viridis colormap
                head_attn_colormap = cv2.applyColorMap(head_attn_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                save_path = os.path.join(vis_token_index_dir, '{}.jpg'.format(head_idx))
                cv2.imwrite(save_path, head_attn_colormap)  # Save the image using cv2.imwrite


if __name__ == '__main__':
    main()
