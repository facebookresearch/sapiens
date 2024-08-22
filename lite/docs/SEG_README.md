# Sapiens-Lite: Body Part Segmentation

## Model Zoo
We use 28 classes for body-part segmentation along with the background class.
You can checkout more details on the classes [here](../../seg/mmseg/datasets/goliath.py).

```
0: Background
1: Apparel
2: Face_Neck
3: Hair
4: Left_Foot
5: Left_Hand
6: Left_Lower_Arm
7: Left_Lower_Leg
8: Left_Shoe
9: Left_Sock
10: Left_Upper_Arm
11: Left_Upper_Leg
12: Lower_Clothing
13: Right_Foot
14: Right_Hand
15: Right_Lower_Arm
16: Right_Lower_Leg
17: Right_Shoe
18: Right_Sock
19: Right_Upper_Arm
20: Right_Upper_Leg
21: Torso
22: Upper_Clothing
23: Lower_Lip
24: Upper_Lip
25: Lower_Teeth
26: Upper_Teeth
27: Tongue
```

The body-part segmentation model checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_$MODE.pt2`
| Sapiens-0.6B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.6b/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_$MODE.pt2`
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_$MODE.pt2`


## Inference Guide

- Navigate to your script directory:
  ```bash
    cd $SAPIENS_LITE_ROOT/scripts/demo/[torchscript,bfloat16]
  ```
- For part segmentation (uncomment your model config line):
  ```bash
  ./seg.sh
  ```

Define `INPUT` for your image directory and `OUTPUT` for results.\
The predictions will be visualized as (.jpg or .png) files, the foreground boolean masks and segmentation probabilities will be stored as .npy files in `OUTPUT` directory.\
These .npy will be used in depth and surface normal visualization.

Adjust `BATCH_SIZE`, `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.\
Note, we skip class label visualization as text on the image in interest of speed.

<p align="center">
  <img src="../assets/seg.gif" alt="Body Part Segmentation" width="800" style="margin-right: 10px;"/>
</p>
