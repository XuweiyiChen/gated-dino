#!/bin/bash

# Script to extract ImageNet class tar files
# Each class tar file will be extracted into its corresponding directory

TRAIN_DIR="/anvil/projects/x-nairr250073/datasets_xuweiyi/imagenet-1k/train"

cd "$TRAIN_DIR" || exit 1

echo "Starting extraction of ImageNet class tar files..."
echo "Directory: $TRAIN_DIR"
echo "Number of tar files: $(ls -1 *.tar 2>/dev/null | wc -l)"

# Counter for progress
count=0
total=$(ls -1 *.tar 2>/dev/null | wc -l)

echo "Starting parallel extraction..."
START_TIME=$(date +%s)

# Extract each tar file into its class directory
for f in *.tar; do
  if [ -f "$f" ]; then
    count=$((count + 1))
    class_dir="${f%.tar}"
    
    echo "[$count/$total] Extracting $f to $class_dir/"
    
    # Extract to class directory (should already exist)
    mkdir -p "$class_dir"
    tar -xf "$f" -C "$class_dir"
    
    # Remove tar file after successful extraction to save space
    if [ $? -eq 0 ]; then
        rm "$f"
        echo "[$count/$total] ✅ Extracted and removed $f"
    else
        echo "[$count/$total] ❌ Error extracting $f"
    fi
  fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================="
echo "Extraction complete!"
echo "Extracted $count class directories"
echo "Time elapsed: $ELAPSED seconds"
echo "========================================="

# Count total images
echo "Counting total images (this may take a moment)..."
total_images=$(find "$TRAIN_DIR" -type f -name "*.JPEG" | wc -l)
echo "Total images: $total_images"

# Show disk usage
echo "Disk usage:"
du -sh "$TRAIN_DIR"

