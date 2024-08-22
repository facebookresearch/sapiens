DATA_ROOT='/mnt/home/rawalk/drive/mmseg/data'
OUTPUT_ROOT='/mnt/home/rawalk/drive/mmseg/Outputs'
CHECKPOINT_ROOT='/mnt/home/rawalk/drive/mmseg/checkpoints'

### cd to root
cd ../..

ln -sfn $DATA_ROOT data
ln -sfn $OUTPUT_ROOT Outputs
ln -sfn $CHECKPOINT_ROOT checkpoints
