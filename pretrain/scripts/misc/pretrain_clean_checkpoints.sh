cd ../..

# ###--------------------------------------------------------------
CHECKPOINT_DIR="/home/${USER}/sapiens_host/pretrain"
python tools/misc/pretrain_clean_checkpoints.py --checkpoint_dir $CHECKPOINT_DIR; exit

# ###--------------------------------------------------------------
# CHECKPOINT_PATH="/home/${USER}/sapiens_host/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth"
# python tools/misc/finetune_clean_checkpoints.py --checkpoint_path $CHECKPOINT_PATH; exit
