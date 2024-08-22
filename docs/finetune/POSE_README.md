# Finetuning Sapiens: 2D Pose Estimation
This guide outlines the process to finetune the pretrained Sapiens model for top-down 2D pose estimation.\
We use two datasets for training as an example.
- [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody): 133 keypoints (17 kps body, 6 kps feet, 68 kps face, 42 kps hands).
- [COCO](https://cocodataset.org/#home): 17 keypoints

## 📂 1. Data Preparation
Set `$DATA_ROOT` as your training data root directory.\
Download the images and 17 kps annotations from [COCO](https://cocodataset.org/#home). Download the 133 kps annotations from [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody). Unzip the images and annotations as subfolders to ```$DATA_ROOT```.\
 Additionally, download the bounding-box detection on the `val2017` set from [COCO_val2017_detections_AP_H_70_person.json](https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_host/pose/person_detection_results) and place it under `$DATA_ROOT/person_detection_results`.

The data directory structure is as follows:

      $DATA_ROOT/
      │   └── train2017
      │   │   └── 000000000009.jpg
      │   │   └── 000000000025.jpg
      │   │   └── 000000000030.jpg
      │   └── val2017
      │   │   └── 000000000139.jpg
      │   │   └── 000000000285.jpg
      │   │   └── 000000000632.jpg
      │   └── annotations
      │   │   └── person_keypoints_train2017.json
      │   │   └── person_keypoints_val2017.json
      │   │   └── coco_wholebody_train_v1.0.json
      │   │   └── coco_wholebody_val_v1.0.json
      │   └── person_detection_results
      │   │   └── COCO_val2017_detections_AP_H_70_person.json


## ⚙️ 2. Configuration Update

Let `$DATASET` be either `coco-wholebody` for 133 kps or `coco` for 17 kps.\
Edit `$SAPIENS_ROOT/pose/configs/sapiens_pose/$DATASET/sapiens_1b-210e_$DATASET-1024x768.py`:

1. Set `pretrained_checkpoint` to your checkpoint path.
2. Update `train_dataloader.dataset.data_root` and `val_dataloader.dataset.data_root` to your `$DATA_ROOT`. eg. ```data/coco```.
3. Update ```val_evaluator.ann_file``` to also point to validation annotation file under `$DATA_ROOT`.
4. Update `bbox_file` to point to the bounding box detection file under `$DATA_ROOT`.

## 🏋️ 3. Finetuning

The following guide is for Sapiens-1B. You can find other backbones to finetune under [pose_configs_133](../../pose/configs/sapiens_pose/coco_wholebody/) and [pose_configs_17](../../pose/configs/sapiens_pose/coco/).\
The training scripts are under: `$SAPIENS_ROOT/pose/scripts/finetune/$DATASET/sapiens_1b`\
Make sure you have activated the sapiens python conda environment.


### A. 🚀 Single-node Training
Use `$SAPIENS_ROOT/pose/scripts/finetune/$DATASET/sapiens_1b/node.sh`.

Key variables:
- `DEVICES`: GPU IDs (e.g., "0,1,2,3,4,5,6,7")
- `TRAIN_BATCH_SIZE_PER_GPU`: Default 2
- `OUTPUT_DIR`: Checkpoint and log directory
- `RESUME_FROM`: Checkpoint to resume training from. Starts training from previous epoch. Defaults to empty string.
- `LOAD_FROM`: Checkpoint to load weight from. Starts training from epoch 0. Defaults to empty string.
- `mode=multi-gpu`: Launch multi-gpu training with multiple workers for dataloading.
- `mode=debug`: (Optional) To debug. Launched single gpu dry run, with single worker for dataloading. Supports interactive debugging with pdb/ipdb.

Note, if you wish to finetune from an existing pose estimation checkpoint, set the `LOAD_FROM` variable.

Launch:
```bash
cd $SAPIENS_ROOT/pose/scripts/finetune/$DATASET/sapiens_1b
./node.sh
  ```

### B. 🌐 Multi-node Training (Slurm)

Use `$SAPIENS_ROOT/pose/scripts/finetune/$DATASET/sapiens_1b/slurm.sh`

Additional variables:
- `CONDA_ENV`: Path to conda environment
- `NUM_NODES`: Number of nodes (default 4, 8 GPUs per node)

Launch:
```bash
cd $SAPIENS_ROOT/pose/scripts/finetune/$DATASET/sapiens_1b
./slurm.sh
  ```
