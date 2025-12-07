# DINOv2 Linear Probing Evaluation

This directory contains code for linear probing evaluation on ImageNet-1k, adapted from the original DINOv2 repository.

## Overview

Linear probing is a standard evaluation protocol for self-supervised learning models:
1. Freeze the pretrained backbone
2. Train a linear classifier on top of frozen features
3. Evaluate classification accuracy

This implementation follows DINOv2's original hyperparameters from Tables 1 and 2 of their paper.

## Usage

### Basic Usage

```bash
python eval/linear_probe/linear_probe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --train-dir /path/to/imagenet/train \
    --val-dir /path/to/imagenet/val \
    --output-dir ./linear_probe_output
```

### Full Options

```bash
python eval/linear_probe/linear_probe.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --train-dir /path/to/imagenet/train \
    --val-dir /path/to/imagenet/val \
    --output-dir ./linear_probe_output \
    --epochs 10 \
    --batch-size 128 \
    --num-workers 8 \
    --epoch-length 1250 \
    --eval-period-iterations 1250 \
    --learning-rates 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 0.1 \
    --n-last-blocks 1 4 \
    --num-classes 1000
```

## Hyperparameters

The script performs a grid search over:
- **Number of blocks**: 1 or 4 last transformer blocks
- **Average pooling**: With or without average pooling of patch tokens
- **Learning rates**: 13 different learning rates (scaled by batch size)

Learning rates are automatically scaled using the linear scaling rule:
```
lr_scaled = lr * (batch_size * num_gpus) / 256.0
```

## Output

The script outputs:
- `results.json`: Best classifier name and accuracy
- Training logs: Progress and intermediate evaluation results

## Reference

Based on DINOv2's original implementation:
- Config: `dinov2/configs/eval/vitg14_pretrain.yaml`
- Code: `dinov2/eval/linear.py`
- Hyperparameters from DINOv2 paper Tables 1 and 2


