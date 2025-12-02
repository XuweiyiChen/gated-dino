#!/usr/bin/env python
"""Inspect the gated checkpoint to see weight structure"""
import torch
import sys

checkpoint_path = "/anvil/scratch/x-xchen8/gated-dino/experiments/dinov2_vitg14_finetune_gate_v2/checkpoints/last.ckpt"

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*80)
print("CHECKPOINT KEYS:")
print("="*80)
for key in ckpt.keys():
    print(f"  - {key}")

print("\n" + "="*80)
print("STATE_DICT STRUCTURE (Top-level modules):")
print("="*80)
state_dict = ckpt['state_dict']
modules = set([k.split('.')[0] for k in state_dict.keys()])
for module in sorted(modules):
    count = len([k for k in state_dict.keys() if k.startswith(module + '.')])
    print(f"  - {module}: {count} parameters")

print("\n" + "="*80)
print("STUDENT BACKBONE STRUCTURE:")
print("="*80)
student_keys = [k for k in state_dict.keys() if k.startswith('student_backbone.')]
print(f"Total student_backbone parameters: {len(student_keys)}")

# Show unique layer types
layers = set(['.'.join(k.split('.')[:3]) for k in student_keys if 'backbone' in k])
for layer in sorted(layers)[:10]:  # Show first 10
    print(f"  - {layer}")

print("\n" + "="*80)
print("GATE PARAMETERS IN STUDENT BACKBONE:")
print("="*80)
gate_keys = [k for k in student_keys if 'gate' in k.lower()]
if gate_keys:
    print(f"Found {len(gate_keys)} gate parameters:")
    for key in gate_keys[:20]:  # Show first 20
        param = state_dict[key]
        print(f"  - {key}: shape={tuple(param.shape)}")
    if len(gate_keys) > 20:
        print(f"  ... and {len(gate_keys) - 20} more")
else:
    print("No gate parameters found!")

print("\n" + "="*80)
print("SAMPLE ATTENTION LAYER KEYS (Block 0):")
print("="*80)
block0_keys = [k for k in student_keys if 'blocks.0.attn' in k]
for key in sorted(block0_keys):
    param = state_dict[key]
    print(f"  - {key}: shape={tuple(param.shape)}")

print("\n" + "="*80)
print("SAMPLE KEYS FROM DIFFERENT BLOCKS:")
print("="*80)
for block_idx in [0, 1, 2, -3, -2, -1]:
    block_keys = [k for k in student_keys if f'blocks.{block_idx}.attn.qkv' in k]
    if block_keys:
        for key in block_keys:
            param = state_dict[key]
            print(f"  - {key}: shape={tuple(param.shape)}")

