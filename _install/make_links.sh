
DEV_DIR=/home/${USER}/Desktop/sapiens/seg
DISK_DIR=/home/${USER}/drive/seg

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints


DEV_DIR=/home/${USER}/Desktop/sapiens/pose
DISK_DIR=/home/${USER}/drive/pose

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints


DEV_DIR=/home/${USER}/Desktop/sapiens/pretrain
DISK_DIR=/home/${USER}/drive/pretrain

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints
