#!/bin/bash

cd ../.. || exit

#----------------------------set your input and output directories----------------------------------------------
VALID_INDEX=0

##-------------------------------------inference-------------------------------------
RUN_FILE='process/vis_video_pose.py'

##------------no visualization flag-----------------
python ${RUN_FILE} \
    --valid_video_index ${VALID_INDEX}
