#!/usr/bin/env bash

# Check that a directory argument is provided
if [ $# -lt 1 ]; then
	echo "Usage: $0 <directory_of_images>"
	exit 1
fi

INPUT_DIR="$1"

# Check if the provided argument is a directory
if [ ! -d "$INPUT_DIR" ]; then
	echo "Error: '$INPUT_DIR' is not a directory."
	exit 1
fi

# Iterate over image files in the specified directory
# Add or remove extensions as needed (e.g., *.jpg, *.jpeg, *.png)
for img in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.png; do
	# If no files match, the loop will try to run on literal "*.jpg" etc.
	# This checks if the file exists
	[ -e "$img" ] || continue

	echo "Processing: $img"
	python magic.py "$img"

	# If you want to break on errors, you can check the exit status:
	if [ $? -ne 0 ]; then
		echo "Error processing $img. Exiting."
		exit 1
	fi
done

echo "All files processed."
