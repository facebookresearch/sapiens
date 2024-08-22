# Sapiens-Lite: Image Encoder

## Model Zoo
Our 1024 x 1024 resolution vision transformers.

| Model         | Checkpoint Path
|---------------|--------------------------------------------------------------------------------------------------
| Sapiens-0.3B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_$MODE.pt2`
| Sapiens-0.6B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_0.6b/sapiens_0.6b_epoch_1600_$MODE.pt2`
| Sapiens-1B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_$MODE.pt2`
| Sapiens-2B  | `$SAPIENS_LITE_CHECKPOINT_ROOT/pretrain/checkpoints/sapiens_2b/sapiens_2b_epoch_660_$MODE.pt2`


## Inference Guide

- Navigate to your script directory:
  ```bash
  cd $SAPIENS_LITE_ROOT/scripts/demo/[torchscript,bfloat16]
  ```
- For image feature extraction (uncomment your model config line):
  ```bash
  ./extract_feature.sh
  ```

Define `INPUT` for your image directory and `OUTPUT` for results. The features are ```C x H x W``` dimensional and saved as .npy files to the `OUTPUT` folder.\
Adjust `BATCH_SIZE`, `JOBS_PER_GPU`, `TOTAL_GPUS` and `VALID_GPU_IDS` for multi-GPU configurations.
