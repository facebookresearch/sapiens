# Sapiens-Lite: Depth Estimation

## Model Zoo

The depth estimation model checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/depth/checkpoints/sapiens_0.3b/sapiens_0.3b_render_people_epoch_100_$MODE.pt2`
| Sapiens-0.6B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/depth/checkpoints/sapiens_0.6b/sapiens_0.6b_render_people_epoch_70_$MODE.pt2`
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/depth/checkpoints/sapiens_1b/sapiens_1b_render_people_epoch_88_$MODE.pt2`
| Sapiens-2B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/depth/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25_$MODE.pt2`

## Inference Guide

- Navigate to your script directory:
  ```bash
    cd $SAPIENS_LITE_ROOT/scripts/demo/[torchscript,bfloat16]
  ```
- For depth estimation (uncomment your model config line):
  ```bash
  ./depth.sh
  ```

Define `INPUT` for your image directory, `SEG_DIR` for the .npy foreground segmentation directory (obtained from body-part segmentation) and `OUTPUT` for results.\
The predictions will be visualized as (.jpg or .png) files to the `OUTPUT` directory as [image, depth as heatmap, surface normal from depth].\
Adjust `BATCH_SIZE`, `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/depth.gif" alt="Depth Prediction" width="1000" style="margin-right: 10px;"/>
</p>
