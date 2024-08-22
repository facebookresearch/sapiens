# Sapiens-Lite: 2D Human Pose Estimation

## Model Zoo
We provide 4 models of varying size. Sapiens-0.3B, Sapiens-0.6B, Sapiens-1B, Sapiens-2B.
In general, performance improves with increasing the model size.


### BBox Detection
We use an offshelf detector to do top-down pose estimation. Please install, download and set the path appropriately.
- Install `mmdet`
  ```bash
  pip install mmdet
  ```
- **RSC Path:**
  ```bash
  /uca/rawalk/sapiens_lite_host/detector/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
  ```
- **DGX Path:**
  ```bash
  /mnt/home/rawalk/sapiens_lite_host/detector/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
  ```
You can also skip using a bounding box detector by remove the `--det-config` and `--det-checkpoint` from the scripts - in this case the entire image is used as input.

### Body: 17 Keypoints
Best for general in-the-wild scenarios with body keypoints only, adhering to the [COCO keypoint format](http://presentations.cocodataset.org/COCO17-Keypoints-Overview.pdf).

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_mpii_crowdpose_aic_best_coco_AP_796_$MODE.pt2`
| Sapiens-0.6B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_mpii_crowdpose_aic_best_coco_AP_812_$MODE.pt2`
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_coco_mpii_crowdpose_aic_best_coco_AP_821_$MODE.pt2`
| Sapiens-2B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_mpii_crowdpose_aic_best_coco_AP_822_$MODE.pt2`


### Body + Face + Hands + Feet: 133 Keypoints
Offers second-best generalization with body, face, hands, and feet keypoints, following the [COCO-WholeBody keypoint format](https://github.com/jin-s13/COCO-WholeBody).

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_$MODE.pt2`
| Sapiens-0.6B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.6b/sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_$MODE.pt2`
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_$MODE.pt2`
| Sapiens-2B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_$MODE.pt2`

### Body + Dense Face + Hands + Feet: 308 Keypoints (Internal)
The highest number of keypoints predictor. Detailed 274 face keypoints. Following the [Sociopticon keypoint format](../../pose/configs/_base_/datasets/goliath.py).

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_1b/sapiens_1b_goliath_coco_wholebody_mpii_crowdpose_aic_best_goliath_AP_640_$MODE.pt2`


## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_LITE_ROOT/scripts/demo/[rsc/dgx]
  ```
- For 17 keypoints estimation (uncomment your model config line for inference):
  ```bash
  ./pose_keypoints17.sh
  ```
- For 133 keypoints estimation (uncomment your model config line for inference):
  ```bash
  ./pose_keypoints133.sh
  ```
- For 308 keypoints estimation (uncomment your model config line for inference, we recommend using face crops for better results!):
  ```bash
  ./pose_keypoints308.sh
  ```
Define `INPUT` for your image directory and `OUTPUT` for results. Visualization and keypoints in JSON format are saved to `OUTPUT`. \
Customize `LINE_THICKNESS`, `RADIUS`, and `KPT_THRES` as needed. Adjust `BATCH_SIZE`, `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations. \
Note, we skip the keypoint skeleton visualization in interest of speed.

<p align="center">
  <img src="../assets/keypoints17.gif" alt="Keypoints 17" width="300" height="600" style="margin-right: 10px;"/>
  <img src="../assets/keypoints133.gif" alt="Keypoints 133" width="300" height="600" style="margin-left: 10px;"/>
  <img src="../assets/keypoints308.gif" alt="Keypoints 308" width="300" height="600" style="margin-left: 10px;"/>
</p>
