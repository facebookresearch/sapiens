cd ../..

# ###--------------------------------------------------------------
CHECKPOINT_DIR="/home/${USER}/sapiens_host/depth"
python tools/misc/finetune_clean_checkpoints.py --checkpoint_dir $CHECKPOINT_DIR; exit

# # ###--------------------------------------------------------------
# CHECKPOINT_PATH="/home/${USER}/sapiens_host/render_people/checkpoints/sapiens_1b/sapiens_1b_render_people_epoch_50.pth"
# python tools/misc/finetune_clean_checkpoints.py --checkpoint_path $CHECKPOINT_PATH; exit
