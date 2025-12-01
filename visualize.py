#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import cv2

import numpy as np
from PIL import Image
from tqdm import tqdm

from sinder import (
    get_tokens,
    load_model,
    load_visual_data,
    pca_array,
)

os.environ['XFORMERS_DISABLED'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument(
        'imgs', nargs='+', type=str, help='path to image/images'
    )
    parser.add_argument(
        '--model', type=str, default='dinov2_vitg14', help='model name'
    )
    parser.add_argument('--workdir', type=str, default='visualize')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/sinder.pt',
        help='Path to checkpoint. Default is checkpoints/sinder.pt (SINDER repaired model)',
    )
    parser.add_argument(
        '--visual_size',
        type=int,
        default=518,
        help='short side size of input image',
    )

    args = parser.parse_args()
    return args


def visualize(args, model, visual_dataset):
    model.eval()

    for d in tqdm(range(len(visual_dataset))):
        visual_image = visual_dataset[d]
        filename = Path(visual_dataset.files[d]).stem
        
        # Skip high_norm.jpg as it's an output image, not input
        if filename == 'high_norm':
            continue
        
        visual_tokens_all = get_tokens(model, visual_image)
        visual_tokens, visual_tokens_cls = zip(*visual_tokens_all)

        # Get original image
        original_img = cv2.imread(str(visual_dataset.files[d]))

        # Generate norm map
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
        
        # Save individual outputs
        cv2.imwrite(str(args.folder / f'{filename}_norm.png'), norm_colored)
        pca_img.save(args.folder / f'{filename}_pca.png')
        
        # Save combined output
        cv2.imwrite(str(args.folder / f'{filename}_combined.png'), combined)


def main():
    args = parse_args()

    args.folder = Path(args.workdir).expanduser()
    os.makedirs(args.folder, exist_ok=True)
    print(args)
    print(' '.join(sys.argv))

    model = load_model(args.model, args.checkpoint)
    visual_dataset = load_visual_data(args, model)
    visualize(args, model, visual_dataset)


if __name__ == '__main__':
    main()
