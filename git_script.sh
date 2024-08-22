#!/bin/sh

if [ $# -eq 0 ]; then
    echo "No commit message provided. Aborting!"
    exit 1
fi

# Create .gitignore if it does not exist
if [ ! -f .gitignore ]; then
    touch .gitignore
fi

# Ignore large files. Note: It appends to .gitignore.
# The loop handles file paths with spaces correctly.
find . -type f -size +100M | sed 's|^\./||' | while IFS= read -r file; do
    grep -qxF "$file" .gitignore || echo "$file" >> .gitignore
done

# Function to add a line to .gitignore if not present
add_to_gitignore() {
    grep -qxF "$1" .gitignore || echo "$1" >> .gitignore
}

# Example of how to include a subdirectory using '!'
# The following line would include the 'list' subdirectory if it existed inside 'mmhuman3d/data'
# add_to_gitignore "!mmhuman3d/data/list"

# Use the function to add lines
add_to_gitignore "pose/data"
add_to_gitignore "pose/Outputs"
add_to_gitignore "pose/checkpoints"

add_to_gitignore "pretrain/data"
add_to_gitignore "pretrain/Outputs"
add_to_gitignore "pretrain/checkpoints"

add_to_gitignore "seg/data"
add_to_gitignore "seg/Outputs"
add_to_gitignore "seg/checkpoints"

add_to_gitignore "__pycache__/"
add_to_gitignore "*.pyc"
add_to_gitignore "*.ipynb_checkpoints"
add_to_gitignore "*.so"
add_to_gitignore "*.DS_Store"
add_to_gitignore "*._*"
add_to_gitignore "*.egg"

# Push using the git command
git add -A
git commit -m "$1"
git push
