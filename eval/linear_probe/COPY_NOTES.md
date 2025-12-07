# Linear Probing Code - Copy Notes

## Source
Copied from DINOv2 original repository:
- Location: `/anvil/scratch/x-xchen8/gated-dino/useless/dinov2`
- Main script: `dinov2/eval/linear.py`
- Config: `dinov2/configs/eval/vitg14_pretrain.yaml`
- Utilities: `dinov2/eval/utils.py`, `dinov2/eval/setup.py`

## Adaptations Made

1. **Model Loading**: Adapted to work with `DINOv2Lightning` checkpoints instead of raw DINOv2 weights
   - Uses `DINOv2Lightning.load_from_checkpoint()` 
   - Accesses backbone via `model.student_backbone.backbone`

2. **Feature Extraction**: Uses `get_intermediate_layers()` method
   - Returns list of `(patch_tokens, cls_token)` tuples
   - Supports extracting last N blocks

3. **Data Loading**: Uses standard `ImageFolder` dataset
   - Compatible with ImageNet directory structure
   - Uses DINOv2's standard transforms

4. **Hyperparameters**: Matches DINOv2 paper Tables 1 & 2
   - 10 epochs
   - 1250 iterations per epoch
   - Learning rate grid search: [1e-5, ..., 0.1]
   - Last blocks: [1, 4]
   - With/without average pooling

## Key Differences from Original

- **Single GPU**: Currently supports single GPU (can be extended)
- **Simplified**: Removed submitit launcher, uses direct execution
- **Checkpoint Format**: Works with PyTorch Lightning checkpoints instead of raw weights

## Usage

See `README.md` for usage instructions.


