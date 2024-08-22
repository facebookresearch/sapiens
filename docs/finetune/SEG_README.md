# Finetuning Sapiens: Body Part Segmentation
This guide outlines the process to finetune the pretrained Sapiens model for body-part segmentation.\
We use the [FaceSynthetics](https://github.com/microsoft/FaceSynthetics) dataset as an example for this tutorial. Other datasets like [LIP](https://sysu-hcp.net/lip/) and [CIHP](https://datasetninja.com/cihp) are also supported.\
Please refer to configurations corresponding to these datasets [here](https://github.com/facebookresearch/sapiens/tree/main/seg/configs/sapiens_seg).


## 📂 1. Data Preparation
Set `$DATA_ROOT` as your training data root directory.\
Download the 100,000 FaceSynthetics samples from [link](https://facesyntheticspubwedata.blob.core.windows.net/iccv-2021/dataset_100000.zip).
Download and unzip the folder ```dataset_100000``` as ```$DATA_ROOT```.

The train data directory structure is as follows:

      $DATA_ROOT/
      │   └── 000000.png
      │   └── 000000_seg.png
      │   └── 000000_ldmks.txt
      |   └── 000001.png
      │   └── 000001_seg.png
      │   └── 000001_ldmks.txt

The downloaded images are as follows:\
-`$DATA_ROOT/*.png`: RGB images.\
-`$DATA_ROOT/*_seg.png`: Segmentation images.\

[FaceSynthetics](https://github.com/microsoft/FaceSynthetics) uses 19 classes for face segmentation along with the background class.

```
0: Background
1: Skin
2: Nose
3: Right_Eye
4: Left_Eye
5: Right_Brow
6: Left_Brow
7: Right_Ear
8: Left_Ear
9: Mouth_Interior
10: Top_Lip
11: Bottom_Lip
12: Neck
13: Hair
14: Beard
15: Clothing
16: Glasses
17: Headwear
18: Facewear
```

## ⚙️ 2. Configuration Update

Edit `$SAPIENS_ROOT/seg/configs/sapiens_seg/seg_face/sapiens_1b_seg_face-1024x768.py`:

1. Set `pretrained_checkpoint` to your checkpoint path.
2. Update `dataset_train.data_root` to your `$DATA_ROOT`. eg. ```data/face/dataset_100000```
3. Update the ```num_classes``` to be equal to number of classes. eg. ```18 + 1 (for background)```
4. (Optional) Adjust hyperparameters like `num_epochs` and `optim_wrapper.optimizer.lr`.
5. (Optional) Update the class names and color palette. Edit ```CLASSES``` and ```PALETTE``` variables in `$SAPIENS_ROOT/seg/mmseg/datasets/face.py`. This file corresponds to the dataset class used by `dataset_train` in the config file.

## 🏋️ 3. Finetuning

The following guide is for Sapiens-1B. Simply choose the config file from [here](../../seg/configs/sapiens_seg/seg_face/) to use other backbones.\
The training scripts are under: `$SAPIENS_ROOT/seg/scripts/finetune/seg_face/sapiens_1b`\
Make sure you have activated the sapiens python conda environment.


### A. 🚀 Single-node Training
Use `$SAPIENS_ROOT/seg/scripts/finetune/seg_face/sapiens_1b/node.sh`.

Key variables:
- `DEVICES`: GPU IDs (e.g., "0,1,2,3,4,5,6,7")
- `TRAIN_BATCH_SIZE_PER_GPU`: Default 2
- `OUTPUT_DIR`: Checkpoint and log directory
- `RESUME_FROM`: Checkpoint to resume training from. Starts training from previous epoch. Defaults to empty string.
- `LOAD_FROM`: Checkpoint to load weight from. Starts training from epoch 0. Defaults to empty string.
- `mode=multi-gpu`: Launch multi-gpu training with multiple workers for dataloading.
- `mode=debug`: (Optional) To debug. Launched single gpu dry run, with single worker for dataloading. Supports interactive debugging with pdb/ipdb.

Note, if you wish to finetune from an existing body-part segmentation checkpoint, set the `LOAD_FROM` variable.

Launch:
```bash
cd $SAPIENS_ROOT/seg/scripts/finetune/seg_face/sapiens_1b
./node.sh
  ```

### B. 🌐 Multi-node Training (Slurm)

Use `$SAPIENS_ROOT/seg/scripts/finetune/seg_face/sapiens_1b/slurm.sh`

Additional variables:
- `CONDA_ENV`: Path to conda environment
- `NUM_NODES`: Number of nodes (default 4, 8 GPUs per node)

Launch:
```bash
cd $SAPIENS_ROOT/seg/scripts/finetune/seg_face/sapiens_1b
./slurm.sh
  ```
