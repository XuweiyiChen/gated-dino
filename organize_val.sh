#!/bin/bash

# Script to organize ImageNet validation images into class folders
# Uses ILSVRC2012_validation_ground_truth.txt

VAL_DIR="/anvil/scratch/x-xchen8/gated-dino/data/imagenet-1k/val"
GT_FILE="/anvil/scratch/x-xchen8/gated-dino/data/imagenet-1k/ILSVRC2012_validation_ground_truth.txt"
TRAIN_DIR="/anvil/scratch/x-xchen8/gated-dino/data/imagenet-1k/train"

echo "Organizing validation images into class folders..."
echo "VAL_DIR: $VAL_DIR"
echo "GT_FILE: $GT_FILE"
echo "TRAIN_DIR: $TRAIN_DIR"

# Get the list of class folders from train directory (sorted)
mapfile -t classes < <(ls -1 "$TRAIN_DIR" | sort)
num_classes=${#classes[@]}

echo "Found $num_classes classes in train directory"
echo "First few classes: ${classes[0]} ${classes[1]} ${classes[2]}"

# Create class directories inside VAL_DIR
for class_name in "${classes[@]}"; do
    mkdir -p "$VAL_DIR/$class_name"
done

echo "Created class directories"

# Read ground truth and move images
image_idx=1
while IFS= read -r label_idx; do
    # Remove any whitespace/newlines
    label_idx=$(echo "$label_idx" | tr -d '[:space:]')
    
    # Skip empty lines
    [ -z "$label_idx" ] && continue
    
    # label_idx is 1-indexed in the file, array is 0-indexed
    array_idx=$((label_idx - 1))
    class_name="${classes[$array_idx]}"
    
    # Format image filename (ILSVRC2012_val_00000001.JPEG, etc.)
    img_file=$(printf "ILSVRC2012_val_%08d.JPEG" $image_idx)
    
    if [ -f "$VAL_DIR/$img_file" ]; then
        mv "$VAL_DIR/$img_file" "$VAL_DIR/$class_name/"
        if [ $((image_idx % 5000)) -eq 0 ]; then
            echo "Processed $image_idx images..."
        fi
    else
        echo "Warning: $img_file not found"
    fi
    
    image_idx=$((image_idx + 1))
done < "$GT_FILE"

echo "Validation images organized into $num_classes class folders!"
echo "Organized $((image_idx - 1)) images total"

