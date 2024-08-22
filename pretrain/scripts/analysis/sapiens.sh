cd ../..

###--------------------------------------------------------------
RUN_FILE='./tools/analysis_tools/get_flops.py'

DATASET='shutterstock_instagram'

##---------------------------------------------------------------
MODEL="mae_sapiens_0.3b-p16_8xb512-coslr-1600e_${DATASET}"
# MODEL="mae_sapiens_0.6b-p16_8xb512-coslr-1600e_${DATASET}"
# MODEL="mae_sapiens_1b-p16_8xb512-coslr-1600e_${DATASET}"
# MODEL="mae_sapiens_2b-p16_8xb512-coslr-1600e_${DATASET}"
# MODEL="mae_sapiens_4b-p16_8xb512-coslr-1600e_${DATASET}"
# MODEL="mae_sapiens_8b-p16_8xb512-coslr-1600e_${DATASET}"

CONFIG_FILE=configs/sapiens_mae/${DATASET}/${MODEL}.py

###--------------------------------------------------------------
python $RUN_FILE $CONFIG_FILE --shape 1024 1024
