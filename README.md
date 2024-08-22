<p align="center">
  <img src="./assets/sapiens_animation.gif" alt="Sapiens" title="Sapiens" width="500"/>
</p>


## Foundation for Human Vision Models
Sapiens offers a comprehensive suite for human-centric vision tasks (e.g., 2D pose, part segmentation, depth, normal, etc.). The model family is pretrained on 300 million in-the-wild human images and shows excellent generalization to unconstrained conditions. These models are also designed for extracting high-resolution features, having been natively trained at a 1024 x 1024 image resolution with a 16-pixel patch size.

## 🚀 Getting Started
On Avatar DGX and RSC the conda environment and checkpoints are already configured.

### Clone the Repository
   ```bash
   git clone git@ghe.oculus-rep.com:rawalk/sapiens.git
   export SAPIENS_ROOT=/path/to/sapiens
   ```

### Set Checkpoint Directory
   -  On DGX
      ```bash
      export SAPIENS_CHECKPOINT_ROOT=/mnt/home/rawalk/sapiens_host
      ```

   -  On RSC
      ```bash
      export SAPIENS_CHECKPOINT_ROOT=/uca/rawalk/sapiens_host
      ```

   - Alternatively, please download the checkpoints from [gdrive](https://drive.google.com/drive/folders/1dAlQ0CLEYbFdGwcDJEaF-g-YHGHCIHqi?usp=drive_link).

      You can be selective about only downloading the checkpoints of interest.

      Set `$SAPIENS_CHECKPOINT_ROOT` to be the path to the `sapiens_host` folder. Checkpoint directory structure:
      ```plaintext
      sapiens_host/
      ├── detector/
      │   └── checkpoints/
      │       └── rtmpose/
      ├── pretrain/
      │   └── checkpoints/
      │       ├── sapiens_0.3b/
      │       ├── sapiens_0.6b/
      │       ├── sapiens_1b/
      │       └── sapiens_2b/
      ├── pose/
      └── seg/
      └── depth/
      └── normal/
      ```

### Activate Pre-configured Environments

   - On DGX101:
      ```bash
      source /mnt/home/rawalk/anaconda3/bin/activate sapiens
      ```

   - On DGX103:
      ```bash
      source /mnt/projects/conda-envs/activate-latest
      ```
      After activation, please follow the trouble-shooting steps (bottom of the page) for user-specific modification.

   - On RSC:
      ```bash
      source /uca/conda-envs/sapiens/bin/activate
      ```

### Recommended: Lite Installation (Inference-only)
   For users setting up their own environment primarily for running existing models in inference mode, we recommend the [Sapiens-Lite installation](lite/README.md).\
   This setup offers optimized inference (4x faster) with minimal dependencies (only pytorch + numpy + cv2). \
   This is recommended for large-scale data processing with our models.

### Full Installation
   To replicate our complete training setup, run the provided installation script. \
   This will create a new conda environment named `sapiens` and install all necessary dependencies.

   ```bash
   cd $SAPIENS_ROOT/_install
   ./conda.sh
   ```

## 🌟 Human-Centric Vision Tasks
We finetune sapiens for multiple human-centric vision tasks. More tasks will be added in the future.
Please checkout the list below.

- ###  🔍 [Image Encoder](docs/PRETRAIN_README.md) <sup><small><a href="lite/docs/PRETRAIN_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### 👤 [Pose Estimation](docs/POSE_README.md) <sup><small><a href="lite/docs/POSE_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### ✂️ [Body Part + Face Segmentation](docs/SEG_README.md) <sup><small><a href="lite/docs/SEG_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### 🔭 [Depth Estimation](docs/DEPTH_README.md) <sup><small><a href="lite/docs/DEPTH_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ###  📏 [Metric Depth Estimation](docs/METRIC_DEPTH_README.md)
- ### 📐 [Surface Normal Estimation](docs/NORMAL_README.md) <sup><small><a href="lite/docs/NORMAL_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ###  🎨 [Albedo Estimation](docs/ALBEDO_README.md)
<!-- ### 🎭 [Face Albedo Estimation](docs/FACE_ALBEDO_README.md) -->
<!-- ### ☀️  [Environment Lighting Estimation](docs/HDRI_README.md) -->
- ### 📦 [Pointmap Estimation](docs/POINTMAP_README.md)

## 🎯 Easy Steps to Finetuning Sapiens
Finetuning our models is super-easy! Here is a detailed training guide for the following tasks.
- ### 📐 [Surface Normal Estimation](docs/finetune/NORMAL_README.md)


## 🛠️ Troubleshooting
  If the sapiens conda environment is broken on DGX/RSC due to rogue modifications, follow these steps for a quick fix (user-specific installation):
  ```bash
  cd $SAPIENS_ROOT/engine; pip install -e .
  cd $SAPIENS_ROOT/cv; pip install -e .
  cd $SAPIENS_ROOT/pretrain; pip install -e .
  cd $SAPIENS_ROOT/pose; pip install -e .
  cd $SAPIENS_ROOT/det; pip install -e .
  cd $SAPIENS_ROOT/seg; pip install -e .
  ```

## 🤝 Support & Contribution
For any questions or issues, please open an issue in the repository or contact the maintainers.
