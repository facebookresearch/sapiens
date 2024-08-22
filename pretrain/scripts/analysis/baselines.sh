cd ../..

###--------------------------------------------------------------
RUN_FILE='./tools/analysis_tools/get_flops.py'


# ##---------------------------------------------------------------
# MODEL="aim_7b-p14"
# CONFIG_FILE=configs/aim/dfn/${MODEL}.py

# ##---------------------------------------------------------------
# MODEL="vit_1b-p14"
# CONFIG_FILE=configs/baselines/vit/${MODEL}.py

##---------------------------------------------------------------
MODEL="vit_6.5b-p14"
CONFIG_FILE=configs/baselines/maws/${MODEL}.py

###--------------------------------------------------------------
python $RUN_FILE $CONFIG_FILE --shape 224 224
