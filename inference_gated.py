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
    
    # Load checkpoint manually to handle the pretrained_checkpoint_path issue
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    
    # Override checkpoint_path to None so it doesn't try to load pretrained weights
    # (the weights are already in the checkpoint's state_dict)
    hparams['checkpoint_path'] = None
    
    print(f"Model: {hparams.get('model_name', 'unknown')}")
    if 'gate_config_dict' in hparams:
        print(f"Gate config: {hparams['gate_config_dict']}")
    
    # Create model with modified hparams
    model = DINOv2Lightning(**hparams)
    
    # Load state dict from checkpoint
    model.load_state_dict(ckpt['state_dict'])
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


class ResizeToMultiple:
    """Resize image so both dimensions are multiples of patch_size, preserving aspect ratio."""
    def __init__(self, short_side=518, patch_size=14):
        self.short_side = short_side
        self.patch_size = patch_size
    
    def __call__(self, img):
        w, h = img.size
        # Resize so short side = short_side
        if h < w:
            new_h = self.short_side
            new_w = int(w * new_h / h)
        else:
            new_w = self.short_side
            new_h = int(h * new_w / w)
        
        # Round to nearest multiple of patch_size
        new_h = (new_h // self.patch_size) * self.patch_size
        new_w = (new_w // self.patch_size) * self.patch_size
        
        return img.resize((new_w, new_h), Image.BICUBIC)


def get_transform(image_size=518, patch_size=14):
    """Get preprocessing transform - resize short side, ensure divisible by patch_size (like visualize.py)."""
    return transforms.Compose([
        ResizeToMultiple(short_side=image_size, patch_size=patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(model, image_tensor, device='cuda', patch_size=14):
    """Extract features from image.
    
    Returns:
        cls_token: [1, embed_dim] - CLS token features
        patch_tokens: [1, num_patches, embed_dim] - Patch token features
        patch_tokens_2d: [H, W, embed_dim] - Reshaped patch tokens
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Calculate grid size from image dimensions
    _, img_h, img_w = image_tensor.shape[1:]  # C, H, W -> get H, W
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    
    # Use model's forward_features directly (handles all the details)
    features = model.forward_features(image_tensor)
    
    # DINOv2 returns dict with different feature types
    if isinstance(features, dict):
        patch_tokens = features.get('x_norm_patchtokens', None)
        cls_token = features.get('x_norm_clstoken', None)
        if patch_tokens is None or cls_token is None:
            # Fallback to prenorm features
            x_full = features.get('x_prenorm', features.get('x'))
            cls_token = x_full[:, 0]
            patch_tokens = x_full[:, 1:]
    else:
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
    
    # Ensure correct shapes: cls_token [1, embed_dim], patch_tokens [1, num_patches, embed_dim]
    if cls_token.dim() == 1:
        cls_token = cls_token.unsqueeze(0)
    
    # Reshape to 2D (H, W may differ for non-square images)
    B, N, C = patch_tokens.shape
    patch_tokens_2d = patch_tokens.reshape(B, grid_h, grid_w, C).squeeze(0)  # [H, W, C]
    
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
        
        # Create visualizations (like visualize.py - preserve aspect ratio)
        original = cv2.imread(img_path)
        
        norm_map = create_norm_map(patch_tokens_2d)
        
        try:
            pca_map = create_pca_map(patch_tokens_2d)
        except ImportError:
            print("  sklearn not available, skipping PCA visualization")
            pca_map = np.zeros_like(norm_map)
        
        # Resize all to same HEIGHT for concatenation (preserve aspect ratio like visualize.py)
        target_height = norm_map.shape[0]
        original_resized = cv2.resize(original, 
                                     (int(original.shape[1] * target_height / original.shape[0]), 
                                      target_height))
        pca_resized = cv2.resize(pca_map, 
                                (int(pca_map.shape[1] * target_height / pca_map.shape[0]), 
                                 target_height))
        
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

