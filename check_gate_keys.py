#!/usr/bin/env python3
"""Quick check for gate keys in checkpoint (memory efficient)"""
import sys
import zipfile
import re

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.ckpt"

print(f"Checking: {checkpoint_path}")

# PyTorch checkpoints are zip files - peek at the structure
try:
    with zipfile.ZipFile(checkpoint_path, 'r') as zf:
        files = zf.namelist()
        
        # Look for QKV weight files and their sizes
        qkv_files = [f for f in files if 'qkv.weight' in f and 'blocks.0.' in f]
        
        if qkv_files:
            print(f"\n✓ Found QKV files for block 0:")
            for f in qkv_files[:2]:  # Just show first 2
                info = zf.getinfo(f)
                print(f"  {f}")
                print(f"    Size: {info.file_size / 1024:.1f} KB")
        
        # Check for hyper_parameters
        hp_file = [f for f in files if 'hyper_parameters' in f or 'hparams' in f]
        if hp_file:
            print(f"\n✓ Hyperparameters file found: {hp_file[0]}")
        
        print(f"\nTotal files in checkpoint: {len(files)}")
        
except Exception as e:
    print(f"Error: {e}")

# Now do a targeted load of just one tensor to check shape
print("\n--- Loading single QKV tensor to verify shape ---")
import torch
import pickle

# Use mmap to avoid loading entire checkpoint
with open(checkpoint_path, 'rb') as f:
    # Seek to check if it's a proper checkpoint
    magic = f.read(2)
    f.seek(0)
    
# Load with mmap_mode to be more memory efficient
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Check state dict
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
    
    # Find first student QKV
    for key in state_dict.keys():
        if 'student_backbone' in key and 'blocks.0' in key and 'qkv.weight' in key:
            shape = state_dict[key].shape
            print(f"\n{key}")
            print(f"  Shape: {shape}")
            
            # Check for gates
            dim = 1536  # ViT-g dim
            expected_no_gate = dim * 3  # 4608
            expected_headwise = dim * 3 + 24  # 4632 (24 heads)
            expected_elementwise = dim * 4  # 6144
            
            if shape[0] == expected_no_gate:
                print(f"  ❌ NO GATES (shape {shape[0]} = {dim}*3)")
            elif shape[0] == expected_headwise:
                print(f"  ✅ HEADWISE GATES PRESENT! (shape {shape[0]} = {dim}*3 + 24 heads)")
            elif shape[0] == expected_elementwise:
                print(f"  ✅ ELEMENTWISE GATES PRESENT! (shape {shape[0]} = {dim}*4)")
            else:
                print(f"  ⚠️ Unknown shape - expected {expected_no_gate}, {expected_headwise}, or {expected_elementwise}")
            break
    
    # Check hyperparameters
    if 'hyper_parameters' in ckpt:
        hp = ckpt['hyper_parameters']
        if 'gate_config_dict' in hp:
            print(f"\n✅ gate_config_dict in hyperparameters:")
            for k, v in hp['gate_config_dict'].items():
                print(f"    {k}: {v}")
        else:
            print(f"\n❌ gate_config_dict NOT in hyperparameters")
            print(f"    Available keys: {list(hp.keys())[:10]}")

print("\nDone!")

