#!/bin/bash

# Directory containing the files
dir="/home/rawalk/Desktop/sapiens"

# Array of directories to ignore
ignore_dirs=(
    "$dir/.git"
    "$dir/pose/data"
    "$dir/pretrain/data"
    "$dir/seg/data"
    "$dir/pose/Outputs"
    "$dir/pretrain/Outputs"
    "$dir/seg/Outputs"
)

# Construct the find command with -not -path options for ignored directories
find_cmd="find $dir -type f"
for ignore_dir in "${ignore_dirs[@]}"; do
    find_cmd+=" -not -path '$ignore_dir/*'"
done

# Execute the find command and process each file
eval "$find_cmd" | while read -r file; do
    # Check if the first line of the file contains the code signature
    if head -n 1 "$file" | grep -q "^# Copyright (c) OpenMMLab. All rights reserved.$"; then
        # Remove the code signature from the file
        sed -i '1d' "$file"
        echo "Removed code signature from $file"
    fi
done
