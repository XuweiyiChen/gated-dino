#!/usr/bin/env python
"""
Visualize DINOv2 finetuned model features from PyTorch Lightning checkpoints.
Usage: python visualize_finetuned.py --checkpoint path/to/checkpoint.ckpt --imgs image1.jpg image2.jpg
"""
import argparse
import os
import sys
from pathlib import Path
import cv2

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA

os.environ['XFORMERS_DISABLED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Finetuned DINOv2')
    parser.add_argument(
        'imgs', nargs='+', type=str, help='path to image/images'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch Lightning checkpoint (.ckpt)',
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='dinov2_vitg14', 
        help='Model architecture name'
    )
    parser.add_argument(
        '--workdir', 
        type=str, 
        default='visualize_output',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--visual_size',
        type=int,
        default=518,
        help='Short side size of input image',
    )
    parser.add_argument(
        '--use_teacher',
        action='store_true',
        help='Use teacher model instead of student'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[-1],
        help='Which layers to visualize (default: -1 for last layer)'
    )

    args = parser.parse_args()
    return args


def load_model_from_checkpoint(model_name, checkpoint_path, use_teacher=False):
    """Load DINOv2 model from PyTorch Lightning checkpoint."""
    print(f'Loading {model_name} from checkpoint: {checkpoint_path}')
    
    # Load the model architecture using torch.hub
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict (Lightning checkpoint has nested structure)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Choose teacher or student
    prefix = 'teacher.' if use_teacher else 'student.backbone.'
    
    # Filter and clean state dict
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove prefix
            new_key = key[len(prefix):]
            model_state_dict[new_key] = value
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f'⚠️  Missing keys: {len(missing_keys)}')
    if unexpected_keys:
        print(f'⚠️  Unexpected keys: {len(unexpected_keys)}')
    
    print(f'✅ Model loaded successfully')
    model = model.cuda()
    model.eval()
    
    return model


def pca_array(tokens, whiten=False):
    """Apply PCA to visualize high-dimensional features as RGB."""
    h, w, c = tokens.shape
    tokens = tokens.detach().cpu()

    pca = PCA(n_components=3, whiten=whiten)
    pca.fit(tokens.reshape(-1, c))
    projected_tokens = pca.transform(tokens.reshape(-1, c))

    t = torch.tensor(projected_tokens)
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)

    array = (normalized_t * 255).byte().numpy()
    array = array.reshape(h, w, 3)

    return Image.fromarray(array).resize((w * 14, h * 14), 0)


def get_tokens(model, image, layers=[-1]):
    """Extract intermediate layer features from the model."""
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).cuda()
        H = image_batch.shape[2]
        W = image_batch.shape[3]
        
        # Get intermediate layers
        # layers is a list of layer indices (negative indexing supported)
        n = len(layers)
        tokens = model.get_intermediate_layers(
            image_batch, n, return_class_token=True, norm=False
        )
        
        # Reshape tokens to (H, W, C)
        tokens = [
            (
                t.reshape(
                    (H // model.patch_size, W // model.patch_size, t.size(-1))
                ),
                tc,
            )
            for t, tc in tokens
        ]

    return tokens


def load_and_preprocess_image(image_path, size=518):
    """Load and preprocess an image for the model."""
    from torchvision import transforms
    
    img = Image.open(image_path).convert('RGB')
    
    # Resize short side to size
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Make dimensions divisible by patch_size (14)
    patch_size = 14
    new_w = (new_w // patch_size) * patch_size
    new_h = (new_h // patch_size) * patch_size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Convert to tensor and normalize (ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    return img_tensor, img


def visualize(args, model, image_paths):
    """Generate visualizations for all images."""
    model.eval()
    output_folder = Path(args.workdir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths, desc='Visualizing'):
        image_path = Path(image_path)
        if not image_path.exists():
            print(f'⚠️  Image not found: {image_path}')
            continue
        
        filename = image_path.stem
        
        # Load and preprocess
        img_tensor, original_img = load_and_preprocess_image(
            str(image_path), args.visual_size
        )
        
        # Get features
        visual_tokens_all = get_tokens(model, img_tensor, args.layers)
        
        for layer_idx, (visual_tokens, visual_tokens_cls) in enumerate(visual_tokens_all):
            # Generate norm map (shows attention magnitude)
            h, w, c = visual_tokens.shape
            norm = ((visual_tokens.norm(dim=-1) / visual_tokens.norm(dim=-1).max()) * 255).byte().cpu().numpy()
            norm_img = Image.fromarray(norm).resize((w * 14, h * 14), 0)
            norm_colored = cv2.applyColorMap(np.array(norm_img), cv2.COLORMAP_JET)

            # Generate PCA map (shows semantic structure)
            pca_img = pca_array(visual_tokens)
            pca_array_cv2 = cv2.cvtColor(np.array(pca_img), cv2.COLOR_RGB2BGR)
            
            # Read original for concatenation
            original_cv2 = cv2.imread(str(image_path))
            
            # Resize all to same height
            target_height = norm_colored.shape[0]
            original_resized = cv2.resize(
                original_cv2, 
                (int(original_cv2.shape[1] * target_height / original_cv2.shape[0]), 
                 target_height)
            )
            pca_resized = cv2.resize(
                pca_array_cv2, 
                (int(pca_array_cv2.shape[1] * target_height / pca_array_cv2.shape[0]), 
                 target_height)
            )
            
            # Concatenate horizontally: Original | Norm | PCA
            combined = np.hstack([original_resized, norm_colored, pca_resized])
            
            # Save outputs
            layer_suffix = f'_layer{args.layers[layer_idx]}' if len(args.layers) > 1 else ''
            cv2.imwrite(str(output_folder / f'{filename}{layer_suffix}_norm.png'), norm_colored)
            pca_img.save(output_folder / f'{filename}{layer_suffix}_pca.png')
            cv2.imwrite(str(output_folder / f'{filename}{layer_suffix}_combined.png'), combined)
            
            print(f'✅ Saved: {filename}{layer_suffix}_combined.png')


def main():
    args = parse_args()
    print(args)
    print(' '.join(sys.argv))
    
    # Load model
    model = load_model_from_checkpoint(
        args.model, 
        args.checkpoint, 
        use_teacher=args.use_teacher
    )
    
    # Visualize
    visualize(args, model, args.imgs)
    
    print(f'\n✅ All visualizations saved to: {args.workdir}')


if __name__ == '__main__':
    main()

