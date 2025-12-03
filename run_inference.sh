#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --account=nairr250073
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/anvil/scratch/x-xchen8/gated-dino/inference_output/slurm_%j.out

cd /anvil/scratch/x-xchen8/gated-dino
source ~/.bashrc
conda activate /anvil/scratch/x-xchen8/gated-dino/env

echo "=== Verifying checkpoint ==="
CKPT="/anvil/scratch/x-xchen8/gated-dino/experiments/dinov2_vitg14_finetune_gate_v3/checkpoints/dinov2_vitg14_epochepoch=00_losstrain/loss=1.06.ckpt"

# Run inference
python inference_gated.py \
    --checkpoint "$CKPT" \
    --images resources/example.jpg resources/villa.png resources/high_norm.jpg \
    --output_dir inference_output \
    --device cuda

echo "=== Done ==="

