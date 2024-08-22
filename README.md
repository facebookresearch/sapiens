<p align="center">
  <img src="./assets/sapiens_animation.gif" alt="Sapiens" title="Sapiens" width="500"/>
</p>


## Foundation for Human Vision Models
Sapiens offers a comprehensive suite for human-centric vision tasks (e.g., 2D pose, part segmentation, depth, normal, etc.). The model family is pretrained on 300 million in-the-wild human images and shows excellent generalization to unconstrained conditions. These models are also designed for extracting high-resolution features, having been natively trained at a 1024 x 1024 image resolution with a 16-pixel patch size.

## 🚀 Getting Started

### Clone the Repository
   ```bash
   git clone git@github.com:facebookresearch/sapiens.git
   export SAPIENS_ROOT=/path/to/sapiens
   ```

### Recommended: Lite Installation (Inference-only)
   For users setting up their own environment primarily for running existing models in inference mode, we recommend the [Sapiens-Lite installation](lite/README.md).\
   This setup offers optimized inference (4x faster) with minimal dependencies (only PyTorch + numpy + cv2). \

### Full Installation
   To replicate our complete training setup, run the provided installation script. \
   This will create a new conda environment named `sapiens` and install all necessary dependencies.

   ```bash
   cd $SAPIENS_ROOT/_install
   ./conda.sh
   ```

   Please download the checkpoints from [gdrive](https://drive.google.com/drive/folders/1dAlQ0CLEYbFdGwcDJEaF-g-YHGHCIHqi?usp=drive_link). You can be selective about only downloading the checkpoints of interest.\
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

## 🌟 Human-Centric Vision Tasks
We finetune sapiens for multiple human-centric vision tasks. Please checkout the list below.

- ###  🔍 [Image Encoder](docs/PRETRAIN_README.md) <sup><small><a href="lite/docs/PRETRAIN_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### 👤 [Pose Estimation](docs/POSE_README.md) <sup><small><a href="lite/docs/POSE_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### ✂️ [Body Part Segmentation](docs/SEG_README.md) <sup><small><a href="lite/docs/SEG_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### 🔭 [Depth Estimation](docs/DEPTH_README.md) <sup><small><a href="lite/docs/DEPTH_README.md" style="color: #FFA500;">[lite]</a></small></sup>
- ### 📐 [Surface Normal Estimation](docs/NORMAL_README.md) <sup><small><a href="lite/docs/NORMAL_README.md" style="color: #FFA500;">[lite]</a></small></sup>

## 🎯 Easy Steps to Finetuning Sapiens
Finetuning our models is super-easy! Here is a detailed training guide for the following tasks.
- ### 📐 [Surface Normal Estimation](docs/finetune/NORMAL_README.md)


## 🛠️ Troubleshooting
  If the sapiens conda environment is broken due to rogue modifications, follow these steps for a quick fix:
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
