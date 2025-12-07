#!/usr/bin/env python3
"""
Linear probing evaluation script for DINOv2 with gated attention.
Adapted from DINOv2's linear evaluation code.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lightly_train_dinov2 import DINOv2Lightning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ImageNet normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_classification_train_transform(
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
):
    """Create training transform for ImageNet classification."""
    transforms_list = [
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation)
    ]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return transforms.Compose(transforms_list)


def make_classification_eval_transform(
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
):
    """Create evaluation transform for ImageNet classification."""
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
    return transforms.Compose(transforms_list)


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    """Create input for linear classifier from intermediate layer outputs."""
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features."""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    """Container for multiple linear classifiers (for grid search over hyperparameters)."""

    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class ModelWithIntermediateLayers(nn.Module):
    """Wrapper to extract intermediate layer outputs from DINOv2."""

    def __init__(self, model, n_last_blocks, train_gates=False, use_grad_checkpointing=True):
        super().__init__()
        self.lightning_model = model
        self.n_last_blocks = n_last_blocks
        self.train_gates = train_gates
        
        # Get backbone reference
        self.backbone = model.student_backbone.backbone
        
        if train_gates:
            # Enable gradient checkpointing to save memory
            if use_grad_checkpointing:
                # Enable gradient checkpointing on backbone blocks
                if hasattr(self.backbone, 'set_grad_checkpointing'):
                    self.backbone.set_grad_checkpointing(True)
                    logger.info("✅ Gradient checkpointing enabled")
                else:
                    # Manual checkpointing setup for DINOv2
                    for block in self.backbone.blocks:
                        block.checkpoint = True
                    logger.info("✅ Manual gradient checkpointing enabled on blocks")
            
            # Freeze everything except gate layers
            for name, param in self.backbone.named_parameters():
                if ".gate." in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # Count trainable gate params
            gate_params = sum(p.numel() for n, p in self.backbone.named_parameters() if ".gate." in n and p.requires_grad)
            logger.info(f"🔓 Gates UNFROZEN - {gate_params:,} trainable gate parameters")
            
            # Set backbone to train mode for gates (but most params frozen)
            self.backbone.train()
        else:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            logger.info("🔒 Backbone fully frozen (including gates)")

    def forward(self, images):
        if self.train_gates:
            # Allow gradients through gates
            features = self.backbone.get_intermediate_layers(
                images, self.n_last_blocks, return_class_token=True, norm=True
            )
        else:
            # No gradients - fully frozen
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True, norm=True
                )
                # Clone to detach from inference mode
                features = [(p.clone(), c.clone()) for p, c in features]
        return features
    
    def get_gate_parameters(self):
        """Return gate parameters for optimizer."""
        if not self.train_gates:
            return []
        return [p for n, p in self.backbone.named_parameters() if ".gate." in n and p.requires_grad]


def scale_lr(learning_rates, batch_size, num_gpus=1):
    """Scale learning rate by batch size (linear scaling rule)."""
    return learning_rates * (batch_size * num_gpus) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000, num_gpus=1):
    """Setup multiple linear classifiers for hyperparameter grid search."""
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size, num_gpus)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                classifier_name = f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                linear_classifiers_dict[classifier_name] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    return linear_classifiers, optim_param_groups


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    num_classes,
    iteration,
    prefixstring="",
    val_to_train_label=None,
):
    """Evaluate all linear classifiers on validation set."""
    logger.info("Running validation!")

    all_logits = {}
    all_targets = None

    feature_model.eval()
    linear_classifiers.eval()
    for classifier_name in linear_classifiers.classifiers_dict.keys():
        all_logits[classifier_name] = []

    for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
        images = images.cuda()
        targets = targets.cuda()
        
        # Map validation labels to training labels if mapping exists
        if val_to_train_label is not None:
            targets = torch.tensor([val_to_train_label[t.item()] for t in targets], device=targets.device)

        features = feature_model(images)
        outputs = linear_classifiers(features)

        if all_targets is None:
            all_targets = targets.cpu().numpy()
        else:
            all_targets = np.concatenate([all_targets, targets.cpu().numpy()], axis=0)

        for classifier_name, logits in outputs.items():
            all_logits[classifier_name].append(logits.cpu())

    # Concatenate all logits
    for classifier_name in linear_classifiers.classifiers_dict.keys():
        all_logits[classifier_name] = torch.cat(all_logits[classifier_name], dim=0)

    # Compute accuracies
    all_targets = all_targets.astype(int)
    results_dict = {}
    max_accuracy = 0
    best_classifier = ""

    for classifier_name in linear_classifiers.classifiers_dict.keys():
        logits = all_logits[classifier_name]
        
        # Top-1 accuracy
        preds = logits.argmax(dim=1).numpy()
        accuracy = (preds == all_targets).mean() * 100.0
        
        # Top-5 accuracy
        top5_preds = torch.topk(logits, k=5, dim=1)[1].numpy()
        top5_accuracy = np.any(top5_preds == all_targets[:, None], axis=1).mean() * 100.0

        logger.info(f"{prefixstring} -- Classifier: {classifier_name} * Top-1: {accuracy:.2f}% Top-5: {top5_accuracy:.2f}%")

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_classifier = classifier_name

    results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}
    logger.info(f"Best classifier: {results_dict['best_classifier']}")

    return results_dict


def train_linear_classifiers(
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    optimizer,
    scheduler,
    epochs,
    epoch_length,
    eval_period_iterations,
    num_classes,
    output_dir,
    use_amp=False,
    val_to_train_label=None,
    use_wandb=False,
):
    """Train linear classifiers."""
    feature_model.eval()  # Feature extractor always in eval mode
    iteration = 0
    max_iter = epochs * epoch_length
    best_accuracy = 0.0
    best_classifier_name = None

    logger.info(f"Starting training for {max_iter} iterations ({epochs} epochs x {epoch_length} iterations)")
    logger.info(f"Number of classifiers: {len(linear_classifiers)}")
    if use_amp:
        logger.info("🔥 Using mixed precision (fp16) training")
    
    # Setup AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Verify classifiers have trainable parameters
    total_params = sum(p.numel() for p in linear_classifiers.parameters())
    trainable_params = sum(p.numel() for p in linear_classifiers.parameters() if p.requires_grad)
    logger.info(f"Classifier params: {trainable_params:,} trainable / {total_params:,} total")

    pbar = tqdm(total=max_iter, desc="Training")

    for epoch in range(epochs):
        linear_classifiers.train()  # Linear classifiers in train mode
        
        for batch_idx, (images, targets) in enumerate(train_data_loader):
            if iteration >= max_iter:
                break

            images = images.cuda()
            targets = targets.cuda()

            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Feature extraction
                features = feature_model(images)
                
                # Classifier forward with grad
                outputs = linear_classifiers(features)

                # Compute losses for all classifiers
                losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, targets) for k, v in outputs.items()}
                loss = sum(losses.values())

            # Backward pass
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Debug: check gradients on first iteration
            if iteration == 0:
                grad_norms = []
                for name, param in linear_classifiers.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                if grad_norms:
                    logger.info(f"First batch gradient norms - min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}, mean: {np.mean(grad_norms):.6f}")
                else:
                    logger.error("❌ No gradients! Something is wrong.")
            
            scheduler.step()

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})

            # Wandb logging (every 10 iterations)
            if use_wandb and iteration % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/iteration": iteration,
                }, step=iteration)

            # Evaluation
            if eval_period_iterations > 0 and (iteration + 1) % eval_period_iterations == 0:
                logger.info(f"\n--- Evaluation at iteration {iteration + 1} ---")
                val_results = evaluate_linear_classifiers(
                    feature_model=feature_model,
                    linear_classifiers=linear_classifiers,
                    data_loader=val_data_loader,
                    num_classes=num_classes,
                    iteration=iteration,
                    prefixstring=f"ITER: {iteration}",
                    val_to_train_label=val_to_train_label,
                )
                if val_results["best_classifier"]["accuracy"] > best_accuracy:
                    best_accuracy = val_results["best_classifier"]["accuracy"]
                    best_classifier_name = val_results["best_classifier"]["name"]
                
                # Wandb validation logging
                if use_wandb:
                    wandb.log({
                        "val/best_accuracy": best_accuracy,
                        "val/best_classifier": best_classifier_name,
                        "val/iteration": iteration,
                    }, step=iteration)
                
                linear_classifiers.train()  # Back to train mode

            iteration += 1

            if batch_idx >= epoch_length - 1:
                break

        if iteration >= max_iter:
            break

    pbar.close()

    # Final evaluation
    logger.info("Running final evaluation...")
    val_results = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        data_loader=val_data_loader,
        num_classes=num_classes,
        iteration=iteration,
        prefixstring="FINAL",
    )

    if val_results["best_classifier"]["accuracy"] > best_accuracy:
        best_accuracy = val_results["best_classifier"]["accuracy"]
        best_classifier_name = val_results["best_classifier"]["name"]

    return val_results, best_classifier_name


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load DINOv2Lightning model from checkpoint.
    
    Handles gated attention checkpoints properly by:
    1. Loading checkpoint manually to access hyperparameters
    2. Setting checkpoint_path=None to prevent re-loading pretrained weights
    3. Loading state_dict from the finetuned checkpoint
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint manually to handle pretrained_checkpoint_path issue
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    
    # Override checkpoint_path to None so it doesn't try to load pretrained weights
    # (the weights are already in the checkpoint's state_dict)
    hparams['checkpoint_path'] = None
    
    logger.info(f"Model: {hparams.get('model_name', 'unknown')}")
    
    # Check for gated attention
    if 'gate_config_dict' in hparams and hparams['gate_config_dict']:
        gate_config = hparams['gate_config_dict']
        logger.info(f"Gate config: enabled={gate_config.get('enabled')}, "
                   f"version={gate_config.get('version')}, "
                   f"headwise={gate_config.get('headwise')}")
    
    # Create model with hyperparameters
    model = DINOv2Lightning(**hparams)
    
    # Load state dict from checkpoint
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    
    # Verify gated attention is loaded
    backbone = model.student_backbone.backbone
    if hasattr(backbone.blocks[0].attn, 'gate'):
        gate = backbone.blocks[0].attn.gate
        if gate is not None:
            logger.info(f"✅ Gated attention V2 loaded - gate shape: {gate.weight.shape}")
    else:
        qkv_shape = backbone.blocks[0].attn.qkv.weight.shape
        if qkv_shape[0] > 4608:  # > 3*1536 means gates present (V1)
            logger.info(f"✅ Gated attention V1 loaded - QKV shape: {qkv_shape}")
        else:
            logger.info(f"No gated attention detected - QKV shape: {qkv_shape}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Linear probing evaluation for DINOv2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to ImageNet training directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to ImageNet validation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./linear_probe_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        default=1250,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        default=1250,
        help="Number of iterations between evaluations",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        help="Learning rates to grid search",
    )
    parser.add_argument(
        "--n-last-blocks",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Number of last blocks to use",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes",
    )
    parser.add_argument(
        "--train-gates",
        action="store_true",
        help="Unfreeze gate layers during linear probing (joint training of gates + linear head)",
    )
    parser.add_argument(
        "--gate-lr",
        type=float,
        default=1e-4,
        help="Learning rate for gate parameters (only used with --train-gates)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision (fp16) training - reduces memory by ~2x",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gated-dino-linear-probe",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name (defaults to checkpoint name)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not installed. Skipping wandb logging.")
    if use_wandb:
        run_name = args.wandb_run_name or os.path.basename(args.checkpoint).replace('.ckpt', '')
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "checkpoint": args.checkpoint,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "epoch_length": args.epoch_length,
                "learning_rates": args.learning_rates,
                "n_last_blocks": args.n_last_blocks,
                "train_gates": args.train_gates,
                "gate_lr": args.gate_lr,
            }
        )
        logger.info(f"📊 Wandb initialized: {args.wandb_project}/{run_name}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device=device)

    # Create feature extractor
    n_last_blocks = max(args.n_last_blocks)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, train_gates=args.train_gates)

    # Create datasets
    train_transform = make_classification_train_transform()
    val_transform = make_classification_eval_transform()

    train_dataset = ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = ImageFolder(args.val_dir, transform=val_transform)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Training classes: {len(train_dataset.classes)}")
    logger.info(f"Validation classes: {len(val_dataset.classes)}")

    # Build label mapping: val_idx -> train_idx
    # This is needed when val is a subset of train with different indices
    val_to_train_label = {}
    for val_idx, val_class in enumerate(val_dataset.classes):
        if val_class in train_dataset.class_to_idx:
            train_idx = train_dataset.class_to_idx[val_class]
            val_to_train_label[val_idx] = train_idx
        else:
            logger.warning(f"Val class {val_class} not found in training set!")
            val_to_train_label[val_idx] = val_idx  # fallback
    
    if len(val_dataset.classes) != len(train_dataset.classes):
        logger.warning(f"⚠️  Class count mismatch! Val has {len(val_dataset.classes)} classes, train has {len(train_dataset.classes)}")
        logger.info(f"Label mapping: val_idx 0 ({val_dataset.classes[0]}) -> train_idx {val_to_train_label[0]}")
        logger.info(f"Label mapping: val_idx 1 ({val_dataset.classes[1]}) -> train_idx {val_to_train_label[1]}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Get sample output to determine feature dimensions
    sample_image, _ = train_dataset[0]
    sample_output = feature_model(sample_image.unsqueeze(0).cuda())
    
    # Debug: print feature shapes
    logger.info(f"Sample output: {len(sample_output)} blocks")
    for i, (patch_tokens, cls_token) in enumerate(sample_output):
        logger.info(f"  Block {i}: patch_tokens={patch_tokens.shape}, cls_token={cls_token.shape}")

    # Setup linear classifiers
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        args.n_last_blocks,
        args.learning_rates,
        args.batch_size,
        args.num_classes,
        num_gpus=1,  # TODO: support multi-GPU
    )

    # Add gate parameters to optimizer if training gates
    if args.train_gates:
        gate_params = feature_model.get_gate_parameters()
        if gate_params:
            optim_param_groups.append({
                "params": gate_params,
                "lr": args.gate_lr,
            })
            logger.info(f"Added gate parameters to optimizer with lr={args.gate_lr}")
        else:
            logger.warning("--train-gates specified but no gate parameters found!")

    # Setup optimizer and scheduler
    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    max_iter = args.epochs * args.epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)

    # Use AMP by default when training gates (saves memory)
    use_amp = args.amp or args.train_gates
    if args.train_gates and not args.amp:
        logger.info("Auto-enabling --amp for --train-gates to save memory")

    # Train
    val_results, best_classifier_name = train_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        eval_period_iterations=args.eval_period_iterations,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        use_amp=use_amp,
        val_to_train_label=val_to_train_label,
        use_wandb=use_wandb,
    )

    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump({
            "best_classifier": best_classifier_name,
            "best_accuracy": val_results["best_classifier"]["accuracy"],
        }, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info(f"Best classifier: {best_classifier_name}")
    logger.info(f"Best accuracy: {val_results['best_classifier']['accuracy']:.2f}%")

    # Finish wandb
    if use_wandb:
        wandb.log({
            "final/best_accuracy": val_results["best_classifier"]["accuracy"],
            "final/best_classifier": best_classifier_name,
        })
        wandb.finish()
        logger.info("📊 Wandb run finished")


if __name__ == "__main__":
    main()

