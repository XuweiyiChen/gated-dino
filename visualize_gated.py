#!/usr/bin/env python
"""Visualize features from a gated DINOv2 model checkpoint.

This script loads a PyTorch Lightning checkpoint containing a finetuned DINOv2
model (potentially with gated attention) and generates visualizations.
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sinder import (
    load_visual_data,
    pca_array,
)

os.environ['XFORMERS_DISABLED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Gated DINOv2 Model')
    parser.add_argument(
        'imgs', nargs='+', type=str, help='path to image/images'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch Lightning checkpoint (.ckpt file)',
    )
    parser.add_argument(
        '--use_teacher',
        action='store_true',
        help='Use teacher backbone instead of student',
    )
    parser.add_argument(
        '--workdir', type=str, default='visualize_gated',
        help='Output directory'
    )
    parser.add_argument(
        '--visual_size',
        type=int,
        default=518,
        help='short side size of input image',
    )
    parser.add_argument(
        '--show_gate_stats',
        action='store_true',
        help='Print gate activation statistics if gates are present',
    )

    args = parser.parse_args()
    return args


def load_gated_model(checkpoint_path, use_teacher=False):
    """Load model from PyTorch Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        use_teacher: If True, extract teacher backbone; else student
        
    Returns:
        model: The backbone model (with or without gates)
        has_gates: Boolean indicating if gates are present
        gate_config: Dict with gate configuration if present
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Check for gates in the checkpoint
    state_dict = ckpt['state_dict']
    sample_qkv_key = 'student_backbone.backbone.blocks.0.attn.qkv.weight'
    if sample_qkv_key in state_dict:
        qkv_shape = state_dict[sample_qkv_key].shape
        expected_normal = 4608  # For ViT-g/14: 3 * 1536
        has_gates = qkv_shape[0] > expected_normal
        print(f"QKV shape: {qkv_shape} - Gates present: {has_gates}")
    else:
        has_gates = False
        print("Warning: Could not determine if gates are present")
    
    # Get gate config from hyperparameters
    gate_config = None
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        gate_config = hparams.get('gate_config_dict')
        if gate_config:
            print(f"Gate configuration: {gate_config}")
    
    # Load the full Lightning module
    from lightly_train_dinov2 import DINOv2Lightning
    
    model = DINOv2Lightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Extract the backbone
    if use_teacher:
        backbone = model.teacher_backbone.backbone
        print("Using teacher backbone")
    else:
        backbone = model.student_backbone.backbone
        print("Using student backbone")
    
    return backbone, has_gates, gate_config


def get_tokens_with_gates(model, visual_image):
    """Extract tokens from model, similar to sinder.get_tokens.
    
    Args:
        model: DINOv2 backbone
        visual_image: Preprocessed image tensor
        
    Returns:
        List of (tokens, cls_token) tuples for each layer
    """
    with torch.no_grad():
        # Forward through patch embedding
        x = model.patch_embed(visual_image.unsqueeze(0))
        
        # Add positional embedding
        x = x + model.pos_embed
        
        # Prepend class token
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        tokens_all = []
        
        # Forward through blocks
        for block in model.blocks:
            x = block(x)
            
            # Extract tokens (remove CLS token)
            tokens = x[:, 1:, :]  # [B, N, C]
            cls_tok = x[:, 0, :]  # [B, C]
            
            # Reshape tokens to [H, W, C] for visualization
            B, N, C = tokens.shape
            H = W = int(N ** 0.5)
            tokens = tokens.reshape(B, H, W, C).squeeze(0)  # [H, W, C]
            
            tokens_all.append((tokens, cls_tok))
        
        return tokens_all


def analyze_gate_activations(model, visual_image):
    """Analyze gate activation patterns if gates are present.
    
    Args:
        model: DINOv2 backbone with gates
        visual_image: Preprocessed image tensor
        
    Returns:
        dict with gate statistics
    """
    gate_stats = {
        'mean_activations': [],
        'std_activations': [],
        'min_activations': [],
        'max_activations': [],
    }
    
    with torch.no_grad():
        x = model.patch_embed(visual_image.unsqueeze(0))
        x = x + model.pos_embed
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        for i, block in enumerate(model.blocks):
            # Check if this block has gated attention
            if hasattr(block.attn, 'gate_dim') and block.attn.gate_dim > 0:
                # Forward through attention to get gate activations
                qkv_output = block.attn.qkv(block.attn.norm1(x) if hasattr(block.attn, 'norm1') else x)
                
                # Extract gate scores (last gate_dim features)
                gate_dim = block.attn.gate_dim
                gate_scores = qkv_output[..., -gate_dim:]
                gate_activations = torch.sigmoid(gate_scores)
                
                gate_stats['mean_activations'].append(gate_activations.mean().item())
                gate_stats['std_activations'].append(gate_activations.std().item())
                gate_stats['min_activations'].append(gate_activations.min().item())
                gate_stats['max_activations'].append(gate_activations.max().item())
            
            x = block(x)
    
    return gate_stats


def visualize(args, model, visual_dataset, has_gates, gate_config):
    """Generate visualizations for each image."""
    model.eval()

    for d in tqdm(range(len(visual_dataset))):
        visual_image = visual_dataset[d]
        filename = Path(visual_dataset.files[d]).stem
        
        # Extract tokens
        visual_tokens_all = get_tokens_with_gates(model, visual_image)
        visual_tokens, visual_tokens_cls = zip(*visual_tokens_all)

        # Get original image
        original_img = cv2.imread(str(visual_dataset.files[d]))

        # Generate norm map (from last layer)
        t = visual_tokens[-1].detach().cpu()
        h, w, c = t.shape
        norm = ((t.norm(dim=-1) / t.norm(dim=-1).max()) * 255).byte().numpy()
        norm_img = Image.fromarray(norm).resize((w * 14, h * 14), 0)
        norm_colored = cv2.applyColorMap(np.array(norm_img), cv2.COLORMAP_JET)

        # Generate PCA map
        pca_img = pca_array(visual_tokens[-1])
        pca_array_cv2 = cv2.cvtColor(np.array(pca_img), cv2.COLOR_RGB2BGR)
        
        # Resize all images to same height for concatenation
        target_height = norm_colored.shape[0]
        original_resized = cv2.resize(original_img, 
                                     (int(original_img.shape[1] * target_height / original_img.shape[0]), 
                                      target_height))
        pca_resized = cv2.resize(pca_array_cv2, 
                                (int(pca_array_cv2.shape[1] * target_height / pca_array_cv2.shape[0]), 
                                 target_height))
        
        # Concatenate horizontally: Original | Norm | PCA
        combined = np.hstack([original_resized, norm_colored, pca_resized])
        
        # Add text overlay indicating if gates are used
        text = f"{'Gated' if has_gates else 'Standard'} DINOv2"
        if has_gates and gate_config:
            gate_type = "Headwise" if gate_config.get('headwise') else "Elementwise"
            text += f" ({gate_type})"
        
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save outputs
        cv2.imwrite(str(args.folder / f'{filename}_norm.png'), norm_colored)
        pca_img.save(args.folder / f'{filename}_pca.png')
        cv2.imwrite(str(args.folder / f'{filename}_combined.png'), combined)
        
        # Analyze gates if requested
        if args.show_gate_stats and has_gates:
            gate_stats = analyze_gate_activations(model, visual_image)
            print(f"\nGate statistics for {filename}:")
            print(f"  Mean activations: {np.mean(gate_stats['mean_activations']):.4f}")
            print(f"  Std activations: {np.mean(gate_stats['std_activations']):.4f}")
            print(f"  Min activation: {np.min(gate_stats['min_activations']):.4f}")
            print(f"  Max activation: {np.max(gate_stats['max_activations']):.4f}")


def main():
    args = parse_args()

    args.folder = Path(args.workdir).expanduser()
    os.makedirs(args.folder, exist_ok=True)
    print(args)
    print(' '.join(sys.argv))

    # Load model from checkpoint
    model, has_gates, gate_config = load_gated_model(
        args.checkpoint, 
        use_teacher=args.use_teacher
    )
    
    # Create a simple args object for load_visual_data
    class SimpleArgs:
        def __init__(self, imgs, visual_size):
            self.imgs = imgs
            self.visual_size = visual_size
    
    simple_args = SimpleArgs(args.imgs, args.visual_size)
    visual_dataset = load_visual_data(simple_args, model)
    
    visualize(args, model, visual_dataset, has_gates, gate_config)
    
    print(f"\nVisualization complete! Results saved to: {args.folder}")
    if not has_gates:
        print("\n⚠️  WARNING: This checkpoint does NOT contain gated attention!")
        print("   The model is using standard DINOv2 attention.")


if __name__ == '__main__':
    main()

