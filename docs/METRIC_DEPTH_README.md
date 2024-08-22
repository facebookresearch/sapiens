# Sapiens: Metric Depth Estimation

## Model Zoo

The metric depth estimation model checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-1B  | `$SAPIENS_CHECKPOINT_ROOT/metric_depth/checkpoints/sapiens_1b/sapiens_1b_metric_render_people_epoch_50.pth`

## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_ROOT/seg/scripts/demo/[rsc/dgx]
  ```
- For depth estimation (uncomment your model config line):
  ```bash
  ./metric_depth.sh
  ```

Define `INPUT` for your image directory, `SEG_DIR` for the .npy foreground segmentation directory (obtained from body-part segmentation) and `OUTPUT` for results.

The predictions will be visualized as (.jpg or .png) files to the `OUTPUT` directory as [image, metric depth as heatmap, surface normal from depth].
The predicted min and max depth (in meters) of the human is visualized as part of the output.

Adjust `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/metric_depth.gif" alt="Metric Depth Prediction" width="1000" style="margin-right: 10px;"/>
</p>
