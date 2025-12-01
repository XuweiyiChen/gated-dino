#!/bin/bash

# Script to extract ImageNet training tar files
# Each class tar file will be extracted into its own directory

TRAIN_DIR="/anvil/scratch/x-xchen8/gated-dino/data/imagenet-1k/train"

cd "$TRAIN_DIR" || exit 1

echo "Starting extraction of ImageNet training data..."
echo "Directory: $TRAIN_DIR"
echo "Number of tar files: $(ls -1 *.tar 2>/dev/null | wc -l)"

# Counter for progress
count=0
total=$(ls -1 *.tar 2>/dev/null | wc -l)

# Extract each tar file
for f in *.tar; do
  if [ -f "$f" ]; then
    count=$((count + 1))
    class_dir="${f%.tar}"
    
    echo "[$count/$total] Extracting $f to $class_dir/"
    
    mkdir -p "$class_dir"
    tar -xf "$f" -C "$class_dir"
    
    # Uncomment the line below to delete tar files after extraction (saves space)
    rm "$f"
  fi
done

echo "Extraction complete! Extracted $count class directories."

