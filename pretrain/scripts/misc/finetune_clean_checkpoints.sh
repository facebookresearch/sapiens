cd ../..

# ###--------------------------------------------------------------
CHECKPOINT_DIR='/uca/rawalk/sapiens_host/depth'
python tools/misc/finetune_clean_checkpoints.py --checkpoint_dir $CHECKPOINT_DIR; exit

# # ###--------------------------------------------------------------
# CHECKPOINT_PATH=/uca/rawalk/sapiens_host/depth/checkpoints/sapiens_2b/sapiens_2b_render_people_epoch_25.pth
# python tools/misc/finetune_clean_checkpoints.py --checkpoint_path $CHECKPOINT_PATH; exit
