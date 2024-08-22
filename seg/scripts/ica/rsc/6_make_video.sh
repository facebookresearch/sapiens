#!/bin/bash

cd ../../.. || exit

#----------------------------set your input and output directories----------------------------------------------
# Set the default value for ICA_DIR
# ICA_DIR='/uca/full_head_ICA/2022-10-19/Chen/Segs/BodySpin/image'
ICA_DIR='/uca/full_head_ICA/2022-10-19/topo_free/01/body'

# If an argument is provided, use it as ICA_DIR
if [ $# -eq 1 ]; then
    ICA_DIR="$1"
fi


cd $ICA_DIR/videos

# Get the width of each video
width_pose=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 pose.mp4)
width_seg=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 seg.mp4)
width_depth=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 depth.mp4)
width_normal=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 normal.mp4)
width_albedo=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 albedo.mp4)


# Use ffmpeg to create the collage
ffmpeg -y -i seg.mp4 -i pose.mp4 -i depth.mp4 -i normal.mp4 -i albedo.mp4 -filter_complex \
"[0:v]crop=$width_pose:ih:0:0[seg1]; \
 [0:v]crop=$width_pose:ih:$width_pose:0[seg2]; \
 [1:v]scale=$width_pose:ih[pose]; \
 [2:v]crop=$width_pose:ih:$width_pose:0[depth]; \
 [3:v]crop=$width_pose:ih:$width_pose:0[normal]; \
 [4:v]crop=$width_pose:ih:$width_pose:0[albedo]; \
 [seg1][pose][seg2][depth][normal][albedo]hstack=inputs=6[v]" \
-map "[v]" -c:v libx264 -crf 18 -preset veryfast collage.mp4

echo "Done! Result stored at:"
realpath $ICA_DIR/videos/collage.mp4
