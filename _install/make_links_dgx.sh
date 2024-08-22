
DEV_DIR=/mnt/home/rawalk/Desktop/sapiens/seg
DISK_DIR=/mnt/home/rawalk/drive/seg

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints


DEV_DIR=/mnt/home/rawalk/Desktop/sapiens/pose
DISK_DIR=/mnt/home/rawalk/drive/pose

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints


DEV_DIR=/mnt/home/rawalk/Desktop/sapiens/pretrain
DISK_DIR=/mnt/home/rawalk/drive/pretrain

echo $DEV_DIR
ln -sfn $DISK_DIR/Outputs $DEV_DIR/Outputs
ln -sfn $DISK_DIR/data $DEV_DIR/data
ln -sfn $DISK_DIR/checkpoints $DEV_DIR/checkpoints
