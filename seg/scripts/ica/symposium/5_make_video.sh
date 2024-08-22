#!/bin/bash

cd ../../.. || exit

#----------------------------set your input and output directories----------------------------------------------
# Set the default value for ICA_DIR

# If an argument is provided, use it as ICA_DIR
if [ $# -eq 1 ]; then
    ICA_DIR="$1"
fi

cd $ICA_DIR/videos

# Get the width of each video
width_pose=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 pose.mp4)

# Use ffmpeg to create the collage
ffmpeg -y -i seg.mp4 -i pose.mp4 -i depth.mp4 -i normal.mp4 -i albedo.mp4 -filter_complex \
"[0:v]crop=$width_pose:ih:$width_pose:0[seg2]; \
 [1:v]scale=$width_pose:ih[pose]; \
 [2:v]crop=$width_pose:ih:$width_pose:0[depth]; \
 [3:v]crop=$width_pose:ih:$width_pose:0[normal]; \
 [4:v]crop=$width_pose:ih:$width_pose:0[albedo]; \
 [pose][seg2][depth][normal][albedo]hstack=inputs=5[v]" \
-map "[v]" -crf 18 -preset slow collage.mp4

echo "Done! Result stored at:"
realpath $ICA_DIR/videos/collage.mp4
