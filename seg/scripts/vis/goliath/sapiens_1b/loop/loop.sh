#!/bin/bash

# Define the start and end indices for the loop
# START_INDEX=1; END_INDEX=8
# START_INDEX=9; END_INDEX=16
# START_INDEX=17; END_INDEX=24
# START_INDEX=25; END_INDEX=31

START_INDEX=32; END_INDEX=34
# START_INDEX=35; END_INDEX=37
# START_INDEX=38; END_INDEX=40
# START_INDEX=41; END_INDEX=42


# Loop over the range defined by START_INDEX and END_INDEX
for i in $(seq $START_INDEX $END_INDEX); do
    # Format the number to two digits
    reel_num=$(printf "reel%02d" "$i")

    # Define the DATA_PREFIX
    DATA_PREFIX="${reel_num}"

    # Call paper_loop.sh with the defined argument
    ./run.sh "${DATA_PREFIX}"

    # Wait for the command to finish before proceeding to the next iteration
    wait
done
