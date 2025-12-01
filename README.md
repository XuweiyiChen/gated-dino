# Official code for SINDER: Repairing the Singular Defects of DINOv2 (ECCV 2024 Oral)

[![🦢 - Paper](https://img.shields.io/badge/🦢-Paper-red)](https://arxiv.org/abs/2407.16826)

![SINDER](./resources/high_norm.jpg)

## Citation

```bibtex
@InProceedings{Haoqi_2024_ECCV,
author = {Wang, Haoqi and Zhang, Tong and Salzmann, Mathieu},
title = {{SINDER}: Repairing the Singular Defects of DINOv2},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2024}
}
```

## Install

```bash
conda env create -f environment.yml
conda activate sinder
pip install -e .
```

## Train

Put ImageNet-1K dataset at `data/imagenet` folder.

```bash
./main.py
```

## Visualize

```bash
# visualize original dinov2
./visualize.py resources/example.jpg

# visualize sinder, ckpt download link below
./visualize.py resources/example.jpg --checkpoint path/to/sinder.pth
```

## Distributed SSL finetuning (Lightly)

Use the bundled PyTorch Lightning script to launch DINOv2-style self-supervised training (requires installing Lightly in editable mode first):

```bash
cd lightly && pip install -e .
cd ..
python lightly_train_dinov2.py \
  --data-dir data/imagenet/train \
  --epochs 50 \
  --batch-size 64 \
  --accelerator gpu \
  --devices 8
```

## Register Tokens (ViT-RGTS)

We vendor the [ViT-RGTS](https://github.com/agorians/vit-rgts) implementation (MIT license) under `registers/` for experiments that require register tokens. Import it via:

```python
from registers import VitRGTS

model = VitRGTS(
    image_size=224,
    patch_size=14,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_register_tokens=4,
)
```

Ensure `einops` is installed (already included in `requirements.txt`).

## Checkpoints

[Google Drive](https://drive.google.com/file/d/1g0Aq5qXYuMmVrN9-gGwC9ybwlCDFAw-l/view?usp=sharing)
