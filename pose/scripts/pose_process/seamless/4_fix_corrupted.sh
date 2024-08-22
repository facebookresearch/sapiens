#!/bin/bash

cd ../.. || exit

#----------------------------set your input and output directories----------------------------------------------
START_VALID_INDEX=0
END_VALID_INDEX=1

##-------------------------------------inference-------------------------------------
RUN_FILE='process/fix_corrupted_video.py'

##------------no visualization flag-----------------
python ${RUN_FILE} \
    --start_valid_index ${START_VALID_INDEX} \
    --end_valid_index ${END_VALID_INDEX} \
