#!/usr/bin/env python
"""
Inference script for gated DINOv2 model.

Usage:
    python inference_gated.py \
        --checkpoint experiments/dinov2_vitg14_finetune_gate_v3/checkpoints/last.ckpt \
        --images /path/to/image1.jpg /path/to/image2.jpg \
        --output_dir inference_output
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lightly_train_dinov2 import DINOv2Lightning


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Gated DINOv2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .ckpt checkpoint file')
    parser.add_argument('--images', nargs='+', type=str, required=True,
                        help='Image paths to process')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Output directory for results')
    parser.add_argument('--use_teacher', action='store_true',
                        help='Use teacher backbone instead of student')
    parser.add_argument('--image_size', type=int, default=518,
                        help='Image size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model(checkpoint_path, use_teacher=False, device='cuda'):
    """Load the gated DINOv2 model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model = DINOv2Lightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    
    # Extract backbone
    if use_teacher:
        backbone = model.teacher_backbone.backbone
        print("Using teacher backbone")
    else:
        backbone = model.student_backbone.backbone
        print("Using student backbone")
    
    # Check if gates are present
    qkv_shape = backbone.blocks[0].attn.qkv.weight.shape
    has_gates = qkv_shape[0] > 4608  # > 3*1536 means gates present
    print(f"QKV shape: {qkv_shape} - Gates present: {has_gates}")
    
    if model.gate_config:
        print(f"Gate config: enabled={model.gate_config.enabled}, "
              f"headwise={model.gate_config.headwise}, "
              f"elementwise={model.gate_config.elementwise}")
    
    return backbone, has_gates


def get_transform(image_size=518):
    """Get preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(model, image_tensor, device='cuda'):
    """Extract features from image.
    
    Returns:
        cls_token: [1, embed_dim] - CLS token features
        patch_tokens: [1, num_patches, embed_dim] - Patch token features
        patch_tokens_2d: [H, W, embed_dim] - Reshaped patch tokens
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Forward through model
    x = model.patch_embed(image_tensor)
    x = x + model.pos_embed
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    
    for block in model.blocks:
        x = block(x)
    
    x = model.norm(x)
    
    cls_token = x[:, 0]  # [1, embed_dim]
    patch_tokens = x[:, 1:]  # [1, num_patches, embed_dim]
    
    # Reshape to 2D
    B, N, C = patch_tokens.shape
    H = W = int(N ** 0.5)
    patch_tokens_2d = patch_tokens.reshape(B, H, W, C).squeeze(0)  # [H, W, C]
    
    return cls_token, patch_tokens, patch_tokens_2d


def create_norm_map(patch_tokens_2d, patch_size=14):
    """Create norm visualization from patch tokens."""
    t = patch_tokens_2d.cpu()
    h, w, c = t.shape
    norm = t.norm(dim=-1)
    norm = ((norm / norm.max()) * 255).byte().numpy()
    norm_img = Image.fromarray(norm).resize((w * patch_size, h * patch_size), Image.NEAREST)
    norm_colored = cv2.applyColorMap(np.array(norm_img), cv2.COLORMAP_JET)
    return norm_colored


def create_pca_map(patch_tokens_2d, patch_size=14):
    """Create PCA visualization from patch tokens."""
    from sklearn.decomposition import PCA
    
    t = patch_tokens_2d.cpu().numpy()
    h, w, c = t.shape
    t_flat = t.reshape(-1, c)
    
    pca = PCA(n_components=3)
    t_pca = pca.fit_transform(t_flat)
    
    # Normalize to 0-255
    t_pca = t_pca - t_pca.min(axis=0)
    t_pca = t_pca / (t_pca.max(axis=0) + 1e-8)
    t_pca = (t_pca * 255).astype(np.uint8)
    
    pca_img = t_pca.reshape(h, w, 3)
    pca_img = cv2.resize(pca_img, (w * patch_size, h * patch_size), interpolation=cv2.INTER_NEAREST)
    
    return pca_img


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, has_gates = load_model(args.checkpoint, args.use_teacher, args.device)
    
    # Get transform
    transform = get_transform(args.image_size)
    
    print(f"\nProcessing {len(args.images)} images...")
    
    for img_path in args.images:
        print(f"\nProcessing: {img_path}")
        filename = Path(img_path).stem
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image)
        
        # Extract features
        cls_token, patch_tokens, patch_tokens_2d = extract_features(model, image_tensor, args.device)
        
        print(f"  CLS token shape: {cls_token.shape}")
        print(f"  Patch tokens shape: {patch_tokens.shape}")
        print(f"  Patch tokens 2D shape: {patch_tokens_2d.shape}")
        
        # Save features
        features = {
            'cls_token': cls_token.cpu(),
            'patch_tokens': patch_tokens.cpu(),
            'image_path': img_path,
            'has_gates': has_gates,
        }
        torch.save(features, os.path.join(args.output_dir, f'{filename}_features.pt'))
        
        # Create visualizations
        original = cv2.imread(img_path)
        original = cv2.resize(original, (args.image_size, args.image_size))
        
        norm_map = create_norm_map(patch_tokens_2d)
        
        try:
            pca_map = create_pca_map(patch_tokens_2d)
        except ImportError:
            print("  sklearn not available, skipping PCA visualization")
            pca_map = np.zeros_like(norm_map)
        
        # Resize all to same size
        h = norm_map.shape[0]
        original_resized = cv2.resize(original, (h, h))
        pca_resized = cv2.resize(pca_map, (h, h))
        
        # Concatenate: Original | Norm | PCA
        combined = np.hstack([original_resized, norm_map, pca_resized])
        
        # Add label
        label = f"{'Gated' if has_gates else 'Standard'} DINOv2"
        cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save
        cv2.imwrite(os.path.join(args.output_dir, f'{filename}_norm.png'), norm_map)
        cv2.imwrite(os.path.join(args.output_dir, f'{filename}_pca.png'), pca_map)
        cv2.imwrite(os.path.join(args.output_dir, f'{filename}_combined.png'), combined)
        
        print(f"  Saved to {args.output_dir}/{filename}_*.png")
    
    print(f"\n✅ Done! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

