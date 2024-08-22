cd ../..

# ###--------------------------------------------------------------
CHECKPOINT_DIR='/uca/rawalk/sapiens_host/'
python tools/misc/clean_checkpoints.py --checkpoint_dir $CHECKPOINT_DIR; exit

###--------------------------------------------------------------
# CHECKPOINT_PATH='/uca/rawalk/sapiens_host/metric_depth/checkpoints/sapiens_1b/sapiens_1b_metric_render_people_epoch_50.pth'
# python tools/misc/clean_checkpoints.py --checkpoint_path $CHECKPOINT_PATH; exit
