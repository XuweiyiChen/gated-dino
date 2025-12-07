#!/usr/bin/env python3
"""
Organize ImageNet validation set into class folders.

The official ImageNet val download is flat:
  val/ILSVRC2012_val_00000001.JPEG
  
This script reorganizes into:
  val_organized/n01440764/ILSVRC2012_val_00000293.JPEG
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
DATA_DIR = Path("/anvil/scratch/x-xchen8/gated-dino/data/imagenet-1k")
VAL_DIR = DATA_DIR / "val"
VAL_ORGANIZED_DIR = DATA_DIR / "val_organized"
GROUND_TRUTH = DATA_DIR / "ILSVRC2012_validation_ground_truth.txt"
TRAIN_DIR = DATA_DIR / "train"

def main():
    # Get synset list from train directory (sorted alphabetically = class order)
    synsets = sorted(os.listdir(TRAIN_DIR))
    assert len(synsets) == 1000, f"Expected 1000 classes, got {len(synsets)}"
    print(f"Found {len(synsets)} synsets from train directory")
    
    # Read ground truth labels (1-indexed)
    with open(GROUND_TRUTH) as f:
        labels = [int(line.strip()) for line in f.readlines()]
    print(f"Found {len(labels)} ground truth labels")
    
    # Create output directories
    VAL_ORGANIZED_DIR.mkdir(exist_ok=True)
    for synset in synsets:
        (VAL_ORGANIZED_DIR / synset).mkdir(exist_ok=True)
    
    # Get all val images
    val_images = sorted([f for f in os.listdir(VAL_DIR) if f.endswith('.JPEG')])
    print(f"Found {len(val_images)} validation images")
    
    if len(val_images) != len(labels):
        print(f"WARNING: Image count ({len(val_images)}) != label count ({len(labels)})")
        print("Will process min of both...")
    
    # Move images to class folders
    moved = 0
    for idx, img_name in enumerate(tqdm(val_images, desc="Organizing")):
        if idx >= len(labels):
            break
            
        # Ground truth is 1-indexed, convert to 0-indexed
        class_idx = labels[idx] - 1
        synset = synsets[class_idx]
        
        src = VAL_DIR / img_name
        dst = VAL_ORGANIZED_DIR / synset / img_name
        
        if not dst.exists():
            shutil.copy2(src, dst)  # copy instead of move to be safe
            moved += 1
    
    print(f"\n✅ Organized {moved} images into {VAL_ORGANIZED_DIR}")
    print(f"You can now use --val-dir {VAL_ORGANIZED_DIR}")

if __name__ == "__main__":
    main()

