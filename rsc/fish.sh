#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/; ## for open3d error

# Function to append alias if it doesn't already exist in the fish configuration
append_alias() {
    local alias_cmd="$1"
    grep -qF "$alias_cmd" ~/.config/fish/config.fish || echo "$alias_cmd" >> ~/.config/fish/config.fish
}

append_function() {
    local func_name="$1"
    local func_cmd="$2"
    # We now check if the function name already exists in the config
    if ! grep -q "function $func_name" ~/.config/fish/config.fish; then
        echo -e "function $func_name\n$func_cmd\nend" >> ~/.config/fish/config.fish
    fi
}

# Adding aliases to fish config
append_alias "alias cd_fm='cd /home/rawalk/Desktop/sapiens'"
append_alias "alias cd_drive='cd /home/rawalk/drive'"

append_alias "alias ..='cd ..'"
append_alias "alias ...='cd ../..'"
append_alias "alias ....='cd ../../..'"
append_alias "alias .....='cd ../../../..'"

append_alias "alias cd_pt='cd /home/rawalk/Desktop/sapiens/pretrain/scripts'"
append_alias "alias cd_p='cd /home/rawalk/Desktop/sapiens/pose/scripts'"
append_alias "alias cd_s='cd /home/rawalk/Desktop/sapiens/seg/scripts'"

append_alias "alias cd_opt='cd /home/rawalk/Desktop/sapiens/pretrain/Outputs'"
append_alias "alias cd_op='cd /home/rawalk/Desktop/sapiens/pose/Outputs'"
append_alias "alias cd_os='cd /home/rawalk/Desktop/sapiens/seg/Outputs'"

append_function "sq" "squeue -u rawalk --format='%i %P %j %.18u %t %M %D %R'"
append_function "sc" "scancel \$argv"
append_function "make_video" "ffmpeg -framerate \$argv[2] -pattern_type glob -i '*.jpg' -pix_fmt yuv420p \$argv[1].mp4"
append_function "make_video_png" "ffmpeg -framerate \$argv[2] -pattern_type glob -i '*.png' -pix_fmt yuv420p \$argv[1].mp4"
append_function "make_video_png_resize" "ffmpeg -framerate \$argv[2] -pattern_type glob -i '*.png' -vf \"scale=1920:1080\" -pix_fmt yuv420p \$argv[1].mp4"

append_function "rp" "realpath \$argv"


# Encrypt and decrypt functions
append_function "encrypt" "set folder \$argv; if test -d \$folder; for file in \$folder/*; if test -f \$file; gpg --symmetric --cipher-algo AES128 --batch --passphrase 'password' --output \$file.gpg \$file; echo \"Done! Encrypted: \$file\"; end; end; end"
append_function "decrypt" "set folder \$argv; if test -d \$folder; for file in \$folder/*.gpg; if test -f \$file; set output (string replace -r '\.gpg\$' '' \$file); gpg --output \$output --batch --passphrase 'password' --decrypt \$file; echo \"Done! Decrypted: \$file\"; end; end; end"

append_function "job" "
    set job_id \$argv[1]
    echo \"Job Information for Job ID: \$job_id\"
    echo \"--------------------------------\"
    scontrol show job \$job_id
"

exec fish
