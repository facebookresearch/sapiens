cd ../..

###--------------------------------------------------------------
RUN_FILE='./tools/analysis_tools/get_flops.py'

DATASET='shutterstock'

##---------------------------------------------------------------
# MODEL="sapiens2_0.3b_${DATASET}"
MODEL="sapiens2_0.6b_${DATASET}"
# MODEL="sapiens2_1b_${DATASET}"
# MODEL="sapiens2_2b_${DATASET}"

CONFIG_FILE=configs/sapiens2_mae/${DATASET}/${MODEL}.py

###--------------------------------------------------------------
python $RUN_FILE $CONFIG_FILE --shape 4096 4096
