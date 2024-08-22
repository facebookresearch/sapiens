DATA_ROOT='/home/rawalk/drive/mmpretrain/data'
OUTPUT_ROOT='/home/rawalk/drive/mmpretrain/Outputs'
CHECKPOINT_ROOT='/home/rawalk/drive/mmpretrain/checkpoints'

### cd to root
cd ../..

ln -sfn $DATA_ROOT data
ln -sfn $OUTPUT_ROOT Outputs
ln -sfn $CHECKPOINT_ROOT checkpoints
