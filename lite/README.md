
<p align="center">
  <img src="../assets/sapiens_lite_logo.png" alt="Sapiens-Lite" title="Sapiens-Lite" width="500"/>
</p>

## ⚡ Introduction
Sapiens-Lite is our optimized "inference-only" solution, offering:

- Up to 4x faster inference
- Minimal dependencies
- Negligible accuracy loss

## 🚀 Getting Started

- Set the sapiens_lite code root.
  ```bash
  export SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite
  ```

- We support lite-inference for multiple GPU architectures, primarily in two modes.
  - `MODE=torchscript`: All GPUs with PyTorch2.2+. Inference at `float32`, slower but closest to original model performance.
  - `MODE=bfloat16`: Optimized mode for A100 GPUs with PyTorch-2.3. Uses [FlashAttention](https://github.com/Dao-AILab/flash-attention) for accelerated inference. Coming Soon!

- Note to Windows users: Please use the python scripts in `./demo` instead of `./scripts`.

- Please download the checkpoints from [hugging-face](https://huggingface.co/facebook/sapiens).\
  Checkpoints are suffixed with "_$MODE.pt2".\
  You can be selective about only downloading the checkpoints of interest.\
  Set `$SAPIENS_LITE_CHECKPOINT_ROOT` to the path of `sapiens_lite_host/$MODE`. Checkpoint directory structure:
  ```plaintext
  sapiens_lite_host/
  ├── torchscript
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
  ├── bfloat16
      ├── pretrain/
      ├── pose/
      └── seg/
      └── depth/
      └── normal/
  ```

## 🔧 Installation
Set up the minimal `sapiens_lite` conda environment (pytorch >= 2.2):
```
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks
```

## 🌟 Sapiens-Lite Inference

Note: For inference in `bfloat16` mode:
- Outputs may result in slight variations from the original `float32` predictions.
- The first model run will `autotune` the model and print the log. Subsequent runs automatically load the tuned model.
- Due to `torch.compile` warmup iterations, you'll observe better speedups with a larger number of images, thanks to amortization.

Available tasks:
- ###  [Image Encoder](docs/PRETRAIN_README.md)
- ### [Pose Estimation](docs/POSE_README.md)
- ### [Body Part Segmentation](docs/SEG_README.md)
- ### [Depth Estimation](docs/DEPTH_README.md)
- ### [Surface Normal Estimation](docs/NORMAL_README.md)


## ⚙️ Converting Models to Lite

Obtain a `torch.ExportedProgram` or `torchscript` from the existing sapiens model checkpoint. Note, this requires the full-install `sapiens` conda env.
```bash
cd $SAPIENS_ROOT/scripts/[pretrain,pose,seg]/optimize/local
./[feature_extracter,keypoints*,seg,depth,normal]_optimizer.sh
```
For inference:
- Use `demo.AdhocImageDataset` wrapped with a `DataLoader` for image fetching and preprocessing.\
- Utilize the `WorkerPool` class for multiprocessing capabilities in tasks like saving predictions and visualizations.
