# Sapiens: Environment Lighting Estimation

## Model Zoo
The hdri estimation checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-1B  | `$SAPIENS_CHECKPOINT_ROOT/hdri/checkpoints/sapiens_1b/sapiens_1b_hdri_epoch_77.pth`

## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_ROOT/seg/scripts/demo/rsc
  ```
- For normal estimation (uncomment your model config line):
  ```bash
  ./hdri.sh
  ```

Define `INPUT` for your image directory and `OUTPUT` for results.

The predictions will be visualized as [image, hdri] (.jpg or .png) and saved as .npy to the `OUTPUT` directory.


Adjust `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/hdri.gif" alt="Environment Lighting Prediction" width="1400" style="margin-right: 10px;"/>
</p>
