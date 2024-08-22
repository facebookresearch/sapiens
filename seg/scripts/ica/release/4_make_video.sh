#!/bin/bash

# Store the original directory
ORIGINAL_DIR=$(pwd)

# If an argument is provided, use it as OUTPUT_DIR
if [ $# -eq 1 ]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR='/home/rawalk/Desktop/sapiens/seg/Outputs/vis/shutterstock_videos/01'
fi

# Navigate to the videos directory
cd "$OUTPUT_DIR/videos" || exit

# Get the width of the pose video (this will be our base width W)
width_pose=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 pose.mp4)

# Use ffmpeg to create the 2x2 collage
ffmpeg -y -i pose.mp4 -i seg.mp4 -i depth.mp4 -i normal.mp4 -filter_complex \
"[0:v]scale=$width_pose:-1[pose]; \
 [1:v]crop=$width_pose:ih:$width_pose:0[seg]; \
 [2:v]crop=$width_pose:ih:$width_pose:0[depth]; \
 [3:v]crop=$width_pose:ih:$width_pose:0[normal]; \
 [pose][seg]hstack=inputs=2[top]; \
 [depth][normal]hstack=inputs=2[bottom]; \
 [top][bottom]vstack=inputs=2[v]" \
-map "[v]" -c:v libx264 -crf 18 -preset veryfast collage.mp4

echo "Done! Result stored at:"
realpath collage.mp4

# Return to the original directory
cd "$ORIGINAL_DIR" || exit
