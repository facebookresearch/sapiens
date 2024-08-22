# Sapiens: Albedo Estimation

## Model Zoo
The albedo estimation checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-1B  | `$SAPIENS_CHECKPOINT_ROOT/albedo/checkpoints/sapiens_1b/sapiens_1b_albedo_render_people_epoch_42.pth`

## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_ROOT/seg/scripts/demo/rsc
  ```
- For albedo estimation (uncomment your model config line):
  ```bash
  ./albedo.sh
  ```

Define `INPUT` for your image directory and `OUTPUT` for results.

The predictions will be visualized as [image, albedo] (.jpg or .png) and saved as .npy to the `OUTPUT` directory.


Adjust `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/albedo.gif" alt="Albedo Prediction" width="800" style="margin-right: 10px;"/>
</p>
