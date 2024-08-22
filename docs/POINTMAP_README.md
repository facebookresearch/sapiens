# Sapiens: Pointmap Estimation

## What is a pointmap?
Pointmap is per-pixel (X, Y, Z) in camera coordinate system.  
To predict pointmap, Sapiens have to reason about 3D geometry of the subject along with the intrinsics of the camera.

## Model Zoo
The pointmap estimation checkpoints are available at,

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-1B  | `$SAPIENS_CHECKPOINT_ROOT/pointmap/checkpoints/sapiens_1b/sapiens_1b_pointmap_render_people_epoch_44.pth`

## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_ROOT/seg/scripts/demo/rsc
  ```
- For pointmap estimation (uncomment your model config line):
  ```bash
  ./pointmap.sh
  ```

Define `INPUT` for your image directory and `OUTPUT` for results.

The predictions will be visualized as [image, pointmap-Z as depth, normal from depth] (.jpg or .png) and saved as .ply to the `OUTPUT` directory.


Adjust `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.

<p align="center">
  <img src="../assets/pointmap.png" alt="Pointmap Prediction" width="800" style="margin-right: 10px;"/>
  <img src="../assets/pointmap.gif" alt="Pointmap Prediction" width="800" style="margin-right: 10px;"/>
</p>
