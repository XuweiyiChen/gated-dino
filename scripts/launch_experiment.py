#!/usr/bin/env python3
"""
Experiment launcher for gated-dino DINOv2 finetuning.

Usage:
    # Create experiment with default settings
    python scripts/launch_experiment.py exp_name configs/dinov2_finetune_gate.yaml
    
    # Create experiment with custom settings
    python scripts/launch_experiment.py dinov2_vitg14_finetune_gate configs/dinov2_finetune_gate.yaml \
        --gpus 4 --time 24:00:00 --batch-size 8
    
    # Then launch:
    cd experiments/exp_name/launch
    sbatch train_slurm.sh
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional


def create_slurm_script(args, exp_dir, config_name, mode="train"):
    """Create SLURM sbatch script"""
    job_name = f"{args.exp_name}_{mode}"
    time_limit = "4:00:00" if mode == "eval" else args.time

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --time={time_limit}
#SBATCH --account={args.account}
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=ALL
#SBATCH --output={exp_dir}/log/slurm_%j.out
#SBATCH --error={exp_dir}/log/slurm_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment: {args.exp_name}"
echo "Mode: {mode}"
echo "Start time: $(date)"

# Set environment variables
export TORCH_HOME=/anvil/projects/x-nairr250073/xuweic/.cache/torch
export TORCH_HUB=/anvil/projects/x-nairr250073/xuweic/.cache/torch/hub
export HF_HOME=/anvil/projects/x-nairr250073/xuweic/.cache/huggingface
export WANDB_DIR={exp_dir}/log

# Change to code directory
cd {exp_dir}/code

# Create results directory
mkdir -p {exp_dir}/checkpoints
mkdir -p {exp_dir}/results/visualizations

"""

    if mode == "train":
        script += f"""
# Run training with PyTorch Lightning (handles DDP internally)
# Batch size is PER GPU, effective = batch_size * {args.gpus}
python lightly_train_dinov2.py --config configs/{config_name}

echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
"""
    else:  # eval mode
        script += f"""
# Run evaluation/visualization
python visualize.py \\
    --checkpoint {exp_dir}/checkpoints/{args.model_name}_finetuned_final.pth \\
    --model {args.model_name} \\
    --folder {exp_dir}/results/visualizations

echo "=============================================="
echo "Evaluation completed at: $(date)"
echo "=============================================="
"""

    return script


def create_local_script(args, exp_dir, config_name, mode="train"):
    """Create local launch script"""
    script = f"""#!/bin/bash
echo "=== Local {mode.upper()} Job ==="
echo "Experiment: {args.exp_name}"
echo "Mode: {mode}"
echo "GPUs: {args.gpus}"
echo "Started at: $(date)"

# Set environment variables
export TORCH_HOME=/anvil/projects/x-nairr250073/xuweic/.cache/torch
export TORCH_HUB=/anvil/projects/x-nairr250073/xuweic/.cache/torch/hub
export HF_HOME=/anvil/projects/x-nairr250073/xuweic/.cache/huggingface
export WANDB_DIR={exp_dir}/log

# Change to code directory
cd {exp_dir}/code

# Create results directory
mkdir -p {exp_dir}/checkpoints
mkdir -p {exp_dir}/results/visualizations

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0
if [ {args.gpus} -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(({args.gpus}-1)))
fi

"""

    if mode == "train":
        script += f"""
# Run training with PyTorch Lightning (handles DDP internally)
# Batch size is PER GPU, effective = batch_size * {args.gpus}
python lightly_train_dinov2.py --config configs/{config_name} 2>&1 | tee {exp_dir}/log/train.log

echo "=============================================="
echo "Training completed at: $(date)"
echo "=============================================="
"""
    else:  # eval mode
        script += f"""
# Run evaluation/visualization
python visualize.py \\
    --checkpoint {exp_dir}/checkpoints/{args.model_name}_finetuned_final.pth \\
    --model {args.model_name} \\
    --folder {exp_dir}/results/visualizations 2>&1 | tee {exp_dir}/log/eval.log

echo "=============================================="
echo "Evaluation completed at: $(date)"
echo "=============================================="
"""

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Launch experiment with organized directory structure"
    )
    parser.add_argument("exp_name", help="Experiment name")
    parser.add_argument("config_file", help="Config file path (relative to project root)")

    # Resource configuration - updated for correct HPC
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument(
        "--time", default="48:00:00", help="Time limit for training (default: 48:00:00)"
    )
    parser.add_argument("--partition", default="ai", help="SLURM partition (default: ai)")
    parser.add_argument(
        "--account", default="nairr250073-ai", help="SLURM account (default: nairr250073-ai)"
    )
    parser.add_argument(
        "--email",
        default="xuweic@email.virginia.edu",
        help="Email for notifications",
    )
    parser.add_argument("--mem", default="128GB", help="Memory per node (default: 128GB)")
    parser.add_argument("--cpus-per-task", type=int, default=64, help="CPUs per task (default: 64)")

    # Training overrides
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (PER GPU)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--model-name", default="dinov2_vitg14", help="Model name (default: dinov2_vitg14)")

    # Flags
    parser.add_argument("--force", action="store_true", help="Overwrite existing experiment")

    args = parser.parse_args()

    # Project paths
    project_root = Path("/anvil/scratch/x-xchen8/gated-dino")
    exp_dir = project_root / "experiments" / args.exp_name

    print(f"=== Setting up experiment: {args.exp_name} ===")
    print(f"Config: {args.config_file}")
    print(f"GPUs: {args.gpus}")
    print(f"Time limit: {args.time}")
    print(f"Memory: {args.mem}")
    print(f"CPUs per task: {args.cpus_per_task}")
    print(f"Partition: {args.partition}")
    print(f"Account: {args.account}")
    print(f"Email: {args.email}")
    print(f"Experiment dir: {exp_dir}")
    if args.batch_size:
        print(f"Batch size (per GPU): {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * args.gpus}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    if args.learning_rate:
        print(f"Learning rate: {args.learning_rate}")

    # Check if experiment exists
    if exp_dir.exists() and not args.force:
        print(f"\nError: Experiment '{args.exp_name}' already exists at {exp_dir}")
        print("Use --force to overwrite")
        sys.exit(1)

    # Create experiment directory structure
    for subdir in ["code", "log", "launch", "checkpoints", "results/visualizations"]:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Copy core code files
    print("\nCopying code files...")
    code_files = ["lightly_train_dinov2.py", "visualize.py"]
    code_dirs = ["models", "configs", "sinder"]

    for file in code_files:
        if (project_root / file).exists():
            shutil.copy2(project_root / file, exp_dir / "code" / file)
            print(f"  Copied {file}")

    for dir_name in code_dirs:
        src_dir = project_root / dir_name
        dst_dir = exp_dir / "code" / dir_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"  Copied {dir_name}/")

    # Create symlinks to data and checkpoints (to avoid copying large files)
    print("\nCreating symlinks...")
    symlinks = [
        ("data", project_root / "data"),
        ("resources", project_root / "resources"),
        ("pretrained_checkpoints", project_root / "checkpoints"),
    ]
    for name, target in symlinks:
        link = exp_dir / "code" / name
        if link.exists() or link.is_symlink():
            link.unlink()
        if target.exists():
            link.symlink_to(target)
            print(f"  Linked {name} -> {target}")

    # Copy and modify config
    print("\nCopying config...")
    config_src = project_root / args.config_file
    config_name = config_src.name

    if not config_src.exists():
        print(f"Error: Config file {config_src} does not exist!")
        sys.exit(1)

    # Read and modify config
    with open(config_src, "r") as f:
        config_content = f.read()

    # Update checkpoint directory
    lines = config_content.split("\n")
    for i, line in enumerate(lines):
        # Update checkpoint dir
        if "dir:" in line and "checkpoint" in lines[max(0, i-5):i+1]:
            lines[i] = f"  dir: {exp_dir}/checkpoints"
        # Update model checkpoint path to use symlink
        if "checkpoint:" in line and "pretrain" in line:
            # Extract just the filename
            parts = line.split(":")
            if len(parts) >= 2:
                orig_path = parts[1].strip()
                if "/" in orig_path:
                    ckpt_name = orig_path.split("/")[-1]
                    lines[i] = f"  checkpoint: pretrained_checkpoints/{ckpt_name}"
        # Update batch size if specified
        if args.batch_size and "batch_size:" in line:
            lines[i] = f"  batch_size: {args.batch_size}                      # PER GPU! ({args.gpus} GPUs x {args.batch_size} = {args.batch_size * args.gpus} effective)"
        # Update epochs if specified
        if args.epochs and "epochs:" in line:
            lines[i] = f"  epochs: {args.epochs}"
        # Update learning rate if specified
        if args.learning_rate and "learning_rate:" in line:
            lines[i] = f"  learning_rate: {args.learning_rate}"
        # Update devices
        if "devices:" in line:
            lines[i] = f"  devices: {args.gpus}                      # Number of GPUs"
        # Update experiment name
        if "name:" in line and "experiment" in lines[max(0, i-3):i]:
            lines[i] = f"  name: {args.exp_name}"
        # Update wandb name
        if "name:" in line and "wandb" in lines[max(0, i-3):i]:
            lines[i] = f"    name: {args.exp_name}"

    config_dst = exp_dir / "code" / "configs" / config_name
    with open(config_dst, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved updated config to configs/{config_name}")

    # Create ALL launch scripts
    print("\nCreating launch scripts...")
    scripts = {}
    script_configs = [("train", "slurm"), ("train", "local"), ("eval", "slurm"), ("eval", "local")]

    for mode, launch in script_configs:
        script_name = f"{mode}_{launch}.sh"
        if launch == "slurm":
            scripts[script_name] = create_slurm_script(args, exp_dir, config_name, mode=mode)
        else:
            scripts[script_name] = create_local_script(args, exp_dir, config_name, mode=mode)

    # Write all scripts
    for script_name, script_content in scripts.items():
        launch_file = exp_dir / "launch" / script_name
        with open(launch_file, "w") as f:
            f.write(script_content)
        launch_file.chmod(0o755)
        print(f"  Created {script_name}")

    # Create experiment README
    readme_content = f"""# Experiment: {args.exp_name}

## Setup

- **Config**: {args.config_file}
- **Model**: {args.model_name}
- **GPUs**: {args.gpus}
- **Batch size (per GPU)**: {args.batch_size or 'from config'}
- **Effective batch size**: {(args.batch_size or 8) * args.gpus}
- **Time limit**: {args.time}
- **Memory**: {args.mem}
- **CPUs per task**: {args.cpus_per_task}
- **Partition**: {args.partition}
- **Account**: {args.account}
- **Email**: {args.email}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure

- `code/`: Snapshot of code and config used for this experiment
- `log/`: SLURM and local output logs
- `launch/`: All launch scripts (train_slurm.sh, train_local.sh, eval_slurm.sh, eval_local.sh)
- `checkpoints/`: Training checkpoints
- `results/`: Evaluation results and visualizations

## Usage

```bash
cd {exp_dir}/launch
```

### Training

**SLURM:**
```bash
sbatch train_slurm.sh
```

**Local:**
```bash
./train_local.sh
```

### Evaluation (uses final checkpoint from training)

**SLURM:**
```bash
sbatch eval_slurm.sh
```

**Local:**
```bash
./eval_local.sh
```

### Monitoring

**SLURM:**
```bash
squeue -u $USER
tail -f {exp_dir}/log/slurm_*.out
```

**Local:**
```bash
tail -f {exp_dir}/log/train.log
```

**Wandb:**
Check project `gated-dino` on wandb.ai

## Files

- Training script: `code/lightly_train_dinov2.py`
- Visualization script: `code/visualize.py`
- Config: `code/configs/{config_name}`
- Checkpoints: `checkpoints/`
- Visualizations: `results/visualizations/`
"""

    with open(exp_dir / "README.md", "w") as f:
        f.write(readme_content)

    print("")
    print("=" * 60)
    print("✅ Experiment setup complete!")
    print("=" * 60)
    print("")
    print("Directory structure:")
    print(f"{exp_dir}/")
    print("├── code/          # Code snapshot")
    print("├── log/           # SLURM/Local logs")
    print("├── launch/        # All launch scripts")
    print("├── checkpoints/   # Training checkpoints")
    print("├── results/       # Evaluation outputs")
    print("└── README.md      # Experiment info")
    print("")
    print("Generated launch scripts:")
    print(f"  {exp_dir}/launch/")
    print("  ├── train_slurm.sh    # Training on SLURM")
    print("  ├── train_local.sh    # Training locally")
    print("  ├── eval_slurm.sh     # Evaluation on SLURM")
    print("  └── eval_local.sh     # Evaluation locally")
    print("")
    print("Usage examples:")
    print(f"  cd {exp_dir}/launch")
    print("  sbatch train_slurm.sh    # Start training on SLURM")
    print("  ./train_local.sh         # Start training locally")
    print("")
    print("Monitor:")
    print("  squeue -u $USER                    # SLURM jobs")
    print(f"  tail -f {exp_dir}/log/slurm_*.out  # SLURM logs")
    print(f"  tail -f {exp_dir}/log/train.log    # Local logs")


if __name__ == "__main__":
    main()
