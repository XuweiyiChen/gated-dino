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

## Gated Attention DINOv2 Finetuning

This repo includes **gated attention** (Qwen3-style) for DINOv2. The gate learns to selectively modulate attention outputs.

### Quick Start (Single GPU)

```bash
python lightly_train_dinov2.py --config configs/dinov2_finetune_gate.yaml
```

### Experiment Management (Recommended for Multi-GPU)

Use the experiment launcher for organized, reproducible experiments:

```bash
# Create experiment with 4 GPUs
python scripts/launch_experiment.py my_gate_exp configs/dinov2_finetune_gate.yaml \
    --gpus 4 --time 24:00:00 --batch-size 16

# Launch on SLURM
cd experiments/my_gate_exp/launch
sbatch train_slurm.sh

# Or run locally
./train_local.sh

# Monitor
tail -f ../log/slurm_*.out
```

### Configuration

Edit `configs/dinov2_finetune_gate.yaml`:

```yaml
gate:
  enabled: true           # Enable gated attention
  headwise: true          # One gate per attention head (24 gates for ViT-G)
  elementwise: false      # One gate per dimension (64 gates per head)
  use_mem_eff: true       # Use xformers/SDPA
```

### CLI Overrides

```bash
# Enable gated attention
python lightly_train_dinov2.py --config configs/dinov2_finetune.yaml --gate-enabled --gate-headwise

# Disable gated attention
python lightly_train_dinov2.py --config configs/dinov2_finetune_gate.yaml --no-gate

# Multi-GPU with precision
python lightly_train_dinov2.py --config configs/dinov2_finetune_gate.yaml \
    --devices 4 --precision bf16-mixed --batch-size 16
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
  --devices 8 \
  --precision bf16-mixed
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

## Inference with Gated DINOv2

After training, run inference on images:

```bash
cd /anvil/scratch/x-xchen8/gated-dino

# Basic inference
python inference_gated.py \
    --checkpoint experiments/dinov2_vitg14_finetune_gate_v3/checkpoints/last.ckpt \
    --images /path/to/image1.jpg /path/to/image2.jpg \
    --output_dir inference_output

# With options
python inference_gated.py \
    --checkpoint experiments/dinov2_vitg14_finetune_gate_v3/checkpoints/last.ckpt \
    --images resources/*.jpg \
    --output_dir inference_output \
    --use_teacher \
    --device cuda
```

### Outputs

For each image:
- `{filename}_features.pt` - CLS token + patch tokens (for downstream tasks)
- `{filename}_norm.png` - Norm visualization (highlights salient regions)
- `{filename}_pca.png` - PCA visualization (semantic regions)
- `{filename}_combined.png` - Original | Norm | PCA side-by-side

## Checkpoints

[Google Drive](https://drive.google.com/file/d/1g0Aq5qXYuMmVrN9-gGwC9ybwlCDFAw-l/view?usp=sharing)
