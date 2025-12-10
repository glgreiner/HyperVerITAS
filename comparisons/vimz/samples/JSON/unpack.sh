#!/bin/sh

# Loop through directory numbers 19 to 25
for dir in $(seq 19 25); do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        
        # Loop through all .tar.xz files in the directory
        for file in "$dir"/*.tar.xz; do
            # Make sure the file actually exists
            [ -e "$file" ] || continue

            echo " Unpacking: $file"
            tar -xJf "$file" -C "$dir"
        done
    else
        echo "Directory $dir does not exist, skipping."
    fi
done