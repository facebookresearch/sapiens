#!/bin/bash

cd ../.. || exit

#----------------------------set your input and output directories----------------------------------------------
GLOBAL_START_INDEX=$1
GLOBAL_END_INDEX=$2

TOTAL_JOBS=16
RUN_FILE='process/convert_valid_videos.py'

##-------------------------------------inference-------------------------------------
RANGE_PER_JOB=$(( (GLOBAL_END_INDEX - GLOBAL_START_INDEX) / TOTAL_JOBS ))
RANGE_PER_JOB=$(( RANGE_PER_JOB > 0 ? RANGE_PER_JOB : 1 ))

for ((i=0; i<TOTAL_JOBS; i++));
do

    START_INDEX=$(( GLOBAL_START_INDEX + i * RANGE_PER_JOB ))
    END_INDEX=$(( START_INDEX + RANGE_PER_JOB ))

    # Make sure the last job covers the remaining range
    if [[ $i -eq $((TOTAL_JOBS - 1)) ]]; then
        END_INDEX=$GLOBAL_END_INDEX
    fi

    # Check if END_INDEX is greater than GLOBAL_END_INDEX
    if [[ $END_INDEX -gt $GLOBAL_END_INDEX ]]; then
        break
    fi

    python ${RUN_FILE} \
        --start_valid_index ${START_INDEX} \
        --end_valid_index ${END_INDEX} &

    sleep 1
done


# Wait for all background processes to finish
wait
