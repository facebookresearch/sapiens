cd ../..

# ###--------------------------------------------------------------
CHECKPOINT_DIR='/uca/rawalk/sapiens_host/pretrain'
python tools/misc/pretrain_clean_checkpoints.py --checkpoint_dir $CHECKPOINT_DIR; exit

# ###--------------------------------------------------------------
# CHECKPOINT_PATH='/uca/rawalk/sapiens_host/pretrain/checkpoints/sapiens_1b/sapiens_1b_shutterstock_instagram_epoch_173_clean.pth'
# python tools/misc/finetune_clean_checkpoints.py --checkpoint_path $CHECKPOINT_PATH; exit