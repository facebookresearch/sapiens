#!/bin/bash

cd ../.. || exit

#----------------------------set your input and output directories----------------------------------------------
START_VALID_INDEX=0
# END_VALID_INDEX=10
END_VALID_INDEX=3000

##-------------------------------------inference-------------------------------------
RUN_FILE='process/convert_valid_videos.py'

##------------no visualization flag-----------------
python ${RUN_FILE} \
    --start_valid_index ${START_VALID_INDEX} \
    --end_valid_index ${END_VALID_INDEX} \
