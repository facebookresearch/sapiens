# Sapiens: Face Albedo Estimation

## Model Zoo
The face albedo estimation checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-1B  | `$SAPIENS_CHECKPOINT_ROOT/albedo/checkpoints/sapiens_1b/sapiens_1b_albedo_epoch_4.pth`

## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_ROOT/seg/scripts/demo/rsc
  ```
- For face albedo estimation (uncomment your model config line):
  ```bash
  ./albedo_face.sh
  ```

Define `INPUT` for your image directory and `OUTPUT` for results.

The predictions will be visualized as [image, albedo] (.jpg or .png) and saved as .npy to the `OUTPUT` directory.


Adjust `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/albedo_face.gif" alt="Face Albedo Prediction" width="800" style="margin-right: 10px;"/>
</p>
