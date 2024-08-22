# Sapiens: Surface Normal Estimation

## Model Zoo
The normal estimation checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.3b/sapiens_0.3b_normal_render_people_epoch_66.pth`
| Sapiens-0.6B  | `$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.6b/sapiens_0.6b_normal_render_people_epoch_200.pth`
| Sapiens-1B  | `$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115.pth`
| Sapiens-2B  | `$SAPIENS_CHECKPOINT_ROOT/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70.pth`

## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_ROOT/seg/scripts/demo/local
  ```
- For normal estimation (uncomment your model config line):
  ```bash
  ./normal.sh
  ```

Define `INPUT` for your image directory, `SEG_DIR` for the .npy foreground segmentation directory (obtained from body-part segmentation) and `OUTPUT` for results.\
The predictions will be visualized as (.jpg or .png) files to the `OUTPUT` directory as [image, surface normal].\
Adjust `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/normal.gif" alt="Normal Prediction" width="800" style="margin-right: 10px;"/>
</p>
