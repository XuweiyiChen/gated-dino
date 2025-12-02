#!/usr/bin/env python
"""Check QKV shapes in detail to see if gates are included"""
import torch

checkpoint_path = "/anvil/scratch/x-xchen8/gated-dino/experiments/dinov2_vitg14_finetune_gate_v2/checkpoints/last.ckpt"

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['state_dict']

print("\n" + "="*80)
print("ALL ATTENTION LAYER PARAMETERS (Block 0):")
print("="*80)
block0_attn = [k for k in state_dict.keys() if 'student_backbone.backbone.blocks.0.attn' in k]
for key in sorted(block0_attn):
    param = state_dict[key]
    print(f"{key}")
    print(f"  Shape: {tuple(param.shape)}")
    print(f"  Dtype: {param.dtype}")
    print()

print("\n" + "="*80)
print("QKV WEIGHT SHAPES FOR FIRST FEW BLOCKS:")
print("="*80)
for i in range(5):
    qkv_weight_key = f'student_backbone.backbone.blocks.{i}.attn.qkv.weight'
    qkv_bias_key = f'student_backbone.backbone.blocks.{i}.attn.qkv.bias'
    
    if qkv_weight_key in state_dict:
        w_shape = state_dict[qkv_weight_key].shape
        b_shape = state_dict[qkv_bias_key].shape if qkv_bias_key in state_dict else None
        print(f"Block {i}:")
        print(f"  qkv.weight: {tuple(w_shape)}")
        if b_shape:
            print(f"  qkv.bias: {tuple(b_shape)}")
        
        # Expected shapes
        hidden_dim = 1536  # ViT-g/14
        num_heads = 24
        expected_normal = 3 * hidden_dim  # 4608
        expected_headwise = 3 * hidden_dim + num_heads  # 4632
        expected_elementwise = 3 * hidden_dim + hidden_dim  # 6144
        
        print(f"  Expected (normal): ({expected_normal}, {hidden_dim})")
        print(f"  Expected (headwise gate): ({expected_headwise}, {hidden_dim})")
        print(f"  Expected (elementwise gate): ({expected_elementwise}, {hidden_dim})")
        print(f"  Actual out_features: {w_shape[0]}")
        
        if w_shape[0] == expected_normal:
            print(f"  ✗ NO GATES (standard attention)")
        elif w_shape[0] == expected_headwise:
            print(f"  ✓ HEADWISE GATES PRESENT")
        elif w_shape[0] == expected_elementwise:
            print(f"  ✓ ELEMENTWISE GATES PRESENT")
        else:
            print(f"  ? UNKNOWN CONFIGURATION")
        print()

print("\n" + "="*80)
print("HYPERPARAMETERS:")
print("="*80)
if 'hyper_parameters' in ckpt:
    hparams = ckpt['hyper_parameters']
    print(f"Keys: {list(hparams.keys())}")
    if 'gate_config' in hparams:
        print(f"Gate config: {hparams['gate_config']}")
    else:
        print("No gate_config in hyperparameters")

