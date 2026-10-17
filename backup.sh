#!/bin/bash

# Define the destination directory in your home folder
BACKUP_DEST="$HOME/3A/Backup"

# Check if the user provided files to back up
if [ "$#" -eq 0 ]; then
    echo "Error: No files specified."
    echo "Usage: ./backup.sh <file_or_folder_1> <file_or_folder_2> ..."
    exit 1
fi

echo "Starting backup to $BACKUP_DEST..."

# Loop through all provided arguments
for item in "$@"; do
    if [ -e "$item" ]; then
        # -a: Archive mode (preserves permissions, times, etc.)
        # -v: Verbose (shows what is being copied)
        # -R: Relative (preserves the directory structure of the copied files)
        rsync -avR "$item" "$BACKUP_DEST/"
    else
        echo "Warning: '$item' does not exist. Skipping."
    fi
done

echo "Backup complete. Safe to exit!"