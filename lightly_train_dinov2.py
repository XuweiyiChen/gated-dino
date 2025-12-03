#!/usr/bin/env python
"""Distributed DINOv2-style SSL finetuning with PyTorch Lightning + Lightly.

Supports loading pretrained DINOv2 weights and finetuning with DINO/iBOT losses.

Example usage (with config):
    python lightly_train_dinov2.py --config configs/dinov2_finetune.yaml

Example usage (with CLI args):
    python lightly_train_dinov2.py \
        --data-dir /anvil/scratch/x-xchen8/gated-dino/data/imagenet-1k/train \
        --checkpoint /anvil/scratch/x-xchen8/gated-dino/checkpoints/dinov2_vitg14_pretrain.pth \
        --model-name dinov2_vitg14 \
        --epochs 10 --batch-size 8 --accelerator gpu --devices 1
"""

import argparse
import copy
import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torchvision import transforms

from lightly.data import LightlyDataset
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead
from lightly.models.utils import (
    random_block_mask,
    update_momentum,
)
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
from lightly.utils.benchmarking import knn_predict


# =============================================================================
# Configuration Dataclasses
# =============================================================================
@dataclass
class ModelConfig:
    name: str = "dinov2_vitg14"
    checkpoint: Optional[str] = None
    ibot_separate_head: bool = False


@dataclass
class GateConfig:
    """Gated attention configuration (Qwen3-style)."""
    enabled: bool = False
    version: int = 1             # V1=fused, V2=split (allows freezing backbone)
    headwise: bool = False       # One gate scalar per attention head
    elementwise: bool = True     # One gate value per dimension (overrides headwise)
    use_mem_eff: bool = True     # Use memory-efficient attention
    freeze_backbone: bool = False  # If True, freeze all except gate layers (V2 only)


@dataclass
class DataConfig:
    train_dir: str = "data/imagenet/train"
    val_dir: Optional[str] = None
    batch_size: int = 32  # PER GPU (effective = batch_size * num_gpus)
    num_workers: int = 8  # Per GPU
    global_crop_size: int = 224
    global_crop_scale: Tuple[float, float] = (0.32, 1.0)
    local_crop_size: int = 98
    local_crop_scale: Tuple[float, float] = (0.05, 0.32)
    n_local_views: int = 8


@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    backbone_lr_scale: float = 0.1
    weight_decay_start: float = 0.04
    weight_decay_end: float = 0.4
    freeze_backbone_epochs: int = 0
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_epochs: int = 30


@dataclass
class DistributedConfig:
    accelerator: str = "gpu"
    devices: str = "auto"
    strategy: str = "ddp_find_unused_parameters_true"
    sync_batchnorm: bool = True
    precision: str = "bf16-mixed"  # "32", "16", "bf16-mixed", etc.


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    save_top_k: int = -1
    save_last: bool = True
    filename: str = None  # If None, use default pattern


@dataclass
class KNNConfig:
    enabled: bool = False
    k: int = 20
    t: float = 0.07
    num_classes: int = 1000
    max_train_samples: int = 50000


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "dinov2-finetune"
    name: Optional[str] = None
    offline: bool = False


@dataclass
class LoggingConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_every_n_steps: int = 10


@dataclass
class ExperimentConfig:
    name: str = "dinov2_finetune"
    seed: int = 42
    notes: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    config = Config()
    
    # Parse model config
    if "model" in yaml_config:
        m = yaml_config["model"]
        config.model = ModelConfig(
            name=m.get("name", config.model.name),
            checkpoint=m.get("checkpoint", config.model.checkpoint),
            ibot_separate_head=m.get("ibot_separate_head", config.model.ibot_separate_head),
        )
    
    # Parse gate config
    if "gate" in yaml_config:
        g = yaml_config["gate"]
        config.gate = GateConfig(
            enabled=g.get("enabled", config.gate.enabled),
            version=g.get("version", config.gate.version),
            headwise=g.get("headwise", config.gate.headwise),
            elementwise=g.get("elementwise", config.gate.elementwise),
            use_mem_eff=g.get("use_mem_eff", config.gate.use_mem_eff),
            freeze_backbone=g.get("freeze_backbone", config.gate.freeze_backbone),
        )
    
    # Parse data config
    if "data" in yaml_config:
        d = yaml_config["data"]
        config.data = DataConfig(
            train_dir=d.get("train_dir", config.data.train_dir),
            val_dir=d.get("val_dir", config.data.val_dir),
            batch_size=d.get("batch_size", config.data.batch_size),
            num_workers=d.get("num_workers", config.data.num_workers),
            global_crop_size=d.get("global_crop_size", config.data.global_crop_size),
            global_crop_scale=tuple(d.get("global_crop_scale", config.data.global_crop_scale)),
            local_crop_size=d.get("local_crop_size", config.data.local_crop_size),
            local_crop_scale=tuple(d.get("local_crop_scale", config.data.local_crop_scale)),
            n_local_views=d.get("n_local_views", config.data.n_local_views),
        )
    
    # Parse training config
    if "training" in yaml_config:
        t = yaml_config["training"]
        config.training = TrainingConfig(
            epochs=t.get("epochs", config.training.epochs),
            learning_rate=t.get("learning_rate", config.training.learning_rate),
            backbone_lr_scale=t.get("backbone_lr_scale", config.training.backbone_lr_scale),
            weight_decay_start=t.get("weight_decay_start", config.training.weight_decay_start),
            weight_decay_end=t.get("weight_decay_end", config.training.weight_decay_end),
            freeze_backbone_epochs=t.get("freeze_backbone_epochs", config.training.freeze_backbone_epochs),
            ema_momentum_start=t.get("ema_momentum_start", config.training.ema_momentum_start),
            ema_momentum_end=t.get("ema_momentum_end", config.training.ema_momentum_end),
            teacher_temp_start=t.get("teacher_temp_start", config.training.teacher_temp_start),
            teacher_temp_end=t.get("teacher_temp_end", config.training.teacher_temp_end),
            teacher_temp_warmup_epochs=t.get("teacher_temp_warmup_epochs", config.training.teacher_temp_warmup_epochs),
        )
    
    # Parse distributed config
    if "distributed" in yaml_config:
        dist = yaml_config["distributed"]
        config.distributed = DistributedConfig(
            accelerator=dist.get("accelerator", config.distributed.accelerator),
            devices=dist.get("devices", config.distributed.devices),
            strategy=dist.get("strategy", config.distributed.strategy),
            sync_batchnorm=dist.get("sync_batchnorm", config.distributed.sync_batchnorm),
            precision=dist.get("precision", config.distributed.precision),
        )
    
    # Parse checkpoint config
    if "checkpoint" in yaml_config:
        ckpt = yaml_config["checkpoint"]
        config.checkpoint = CheckpointConfig(
            dir=ckpt.get("dir", config.checkpoint.dir),
            save_every_n_epochs=ckpt.get("every_n_epochs", ckpt.get("save_every_n_epochs", config.checkpoint.save_every_n_epochs)),
            save_top_k=ckpt.get("save_top_k", config.checkpoint.save_top_k),
            save_last=ckpt.get("save_last", config.checkpoint.save_last),
            filename=ckpt.get("filename", config.checkpoint.filename),
        )
    
    # Parse KNN config
    if "knn" in yaml_config:
        knn = yaml_config["knn"]
        config.knn = KNNConfig(
            enabled=knn.get("enabled", config.knn.enabled),
            k=knn.get("k", config.knn.k),
            t=knn.get("t", config.knn.t),
            num_classes=knn.get("num_classes", config.knn.num_classes),
            max_train_samples=knn.get("max_train_samples", config.knn.max_train_samples),
        )
    
    # Parse logging config
    if "logging" in yaml_config:
        log = yaml_config["logging"]
        wandb_cfg = log.get("wandb", {})
        config.logging = LoggingConfig(
            wandb=WandbConfig(
                enabled=wandb_cfg.get("enabled", config.logging.wandb.enabled),
                project=wandb_cfg.get("project", config.logging.wandb.project),
                name=wandb_cfg.get("name", config.logging.wandb.name),
                offline=wandb_cfg.get("offline", config.logging.wandb.offline),
            ),
            log_every_n_steps=log.get("log_every_n_steps", config.logging.log_every_n_steps),
        )
    
    # Parse experiment config
    if "experiment" in yaml_config:
        exp = yaml_config["experiment"]
        config.experiment = ExperimentConfig(
            name=exp.get("name", config.experiment.name),
            seed=exp.get("seed", config.experiment.seed),
            notes=exp.get("notes", config.experiment.notes),
        )
    
    return config


def config_to_flat_dict(config: Config) -> dict:
    """Convert nested config to flat dict for logging."""
    flat = {}
    for section_name in ["model", "gate", "data", "training", "distributed", "checkpoint", "knn", "experiment"]:
        section = getattr(config, section_name)
        for key, value in section.__dict__.items():
            flat[f"{section_name}.{key}"] = value
    flat["logging.log_every_n_steps"] = config.logging.log_every_n_steps
    flat["logging.wandb.enabled"] = config.logging.wandb.enabled
    flat["logging.wandb.project"] = config.logging.wandb.project
    return flat


# DINOv2 model configurations
DINOV2_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14, "num_heads": 6, "depth": 12},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14, "num_heads": 12, "depth": 12},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14, "num_heads": 16, "depth": 24},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14, "num_heads": 24, "depth": 40},
}


def freeze_eval_module(module: Module) -> None:
    """Freeze module parameters and set eval mode."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def load_dinov2_backbone(
    model_name: str,
    checkpoint_path: str = None,
    gate_config: GateConfig = None,
):
    """Load DINOv2 backbone from torch.hub or local checkpoint.
    
    Args:
        model_name: DINOv2 model name (e.g., dinov2_vitg14)
        checkpoint_path: Path to local pretrained weights
        gate_config: Gate configuration for gated attention
    
    Returns:
        DINOv2 backbone model, optionally with gated attention
    """
    # Load architecture from torch.hub (no weights)
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=False,  # Don't download weights
    )
    
    # Load weights from local checkpoint if provided
    if checkpoint_path:
        print(f"Loading pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle different checkpoint formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Load with strict=False to handle minor mismatches
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
    
    # Apply gated attention if enabled
    if gate_config is not None and gate_config.enabled:
        version = getattr(gate_config, 'version', 1)
        print(f"Replacing attention with gated attention V{version} (headwise={gate_config.headwise}, elementwise={gate_config.elementwise})")
        
        if version == 2:
            from models.gated_attention_v2 import replace_attention_with_gated_v2
            model = replace_attention_with_gated_v2(
                model,
                headwise_gate=gate_config.headwise,
                elementwise_gate=gate_config.elementwise,
                use_mem_eff=gate_config.use_mem_eff,
            )
        else:
            from models.gated_attention import replace_attention_with_gated
            model = replace_attention_with_gated(
                model,
                headwise_gate=gate_config.headwise,
                elementwise_gate=gate_config.elementwise,
                use_mem_eff=gate_config.use_mem_eff,
            )
        
        # Freeze backbone if requested (V2 only)
        if getattr(gate_config, 'freeze_backbone', False):
            if version != 2:
                print("WARNING: freeze_backbone requires version=2, ignoring")
            else:
                print("Freezing backbone, training only gate layers...")
                for name, param in model.named_parameters():
                    if ".gate." in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                
                # Count trainable params
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
        
        # Count gate parameters
        gate_params = sum(
            p.numel() for n, p in model.named_parameters() 
            if (".gate." in n or "qkv" in n) and p.requires_grad
        )
        print(f"Gate-related trainable parameters: {gate_params:,}")
    
    return model


class DINOv2Wrapper(nn.Module):
    """Wrapper around DINOv2 backbone for Lightly compatibility."""
    
    def __init__(self, backbone: nn.Module, patch_size: int = 14):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self._sequence_length = None
    
    @property
    def sequence_length(self):
        """Return sequence length (num_patches + 1 for CLS)."""
        if self._sequence_length is None:
            # Default for 224x224 images with patch_size=14: (224/14)^2 + 1 = 257
            self._sequence_length = (224 // self.patch_size) ** 2 + 1
        return self._sequence_length
    
    @property
    def grid_size(self):
        """Return (H, W) grid size."""
        h = w = 224 // self.patch_size
        return (h, w)
    
    def encode(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Encode images to patch features."""
        # DINOv2 forward_features returns dict with keys: x_norm_clstoken, x_norm_patchtokens, etc.
        # We need the full features including CLS token
        features = self.backbone.forward_features(x)
        
        # Concatenate CLS token with patch tokens
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken", features.get("x_prenorm")[:, 0:1])
            patch_tokens = features.get("x_norm_patchtokens", features.get("x_prenorm")[:, 1:])
            all_features = torch.cat([cls_token.unsqueeze(1) if cls_token.dim() == 2 else cls_token, patch_tokens], dim=1)
        else:
            all_features = features
        
        return all_features
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning CLS token."""
        return self.backbone(x)


class DINOv2Head(Module):
    """Utility container that bundles the DINO/iBOT heads."""

    def __init__(
        self,
        dino_head: DINOv2ProjectionHead,
        ibot_head: DINOv2ProjectionHead,
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2Lightning(pl.LightningModule):
    """LightningModule for DINOv2 SSL finetuning."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        checkpoint_path: str = None,
        ibot_separate_head: bool = False,
        base_lr: float = 1e-4,  # Lower LR for finetuning
        freeze_backbone_epochs: int = 0,  # Optionally freeze backbone initially
        gate_config: GateConfig = None,  # Gated attention config (for fresh training)
        gate_config_dict: dict = None,  # For loading from checkpoint
    ) -> None:
        super().__init__()
        
        # If gate_config_dict is provided (from checkpoint loading), reconstruct GateConfig
        if gate_config is None and gate_config_dict is not None:
            gate_config = GateConfig(**gate_config_dict)
            print(f"Reconstructed gate_config from checkpoint: {gate_config}")
        
        # Convert gate_config to dict for serialization
        if gate_config is not None:
            gate_config_dict = {
                'enabled': gate_config.enabled,
                'version': getattr(gate_config, 'version', 1),
                'headwise': gate_config.headwise,
                'elementwise': gate_config.elementwise,
                'use_mem_eff': gate_config.use_mem_eff,
                'freeze_backbone': getattr(gate_config, 'freeze_backbone', False),
            }
        
        # Save hyperparameters (gate_config_dict will be saved, not gate_config object)
        self.save_hyperparameters(ignore=["gate_config"])
        self.gate_config = gate_config  # Store for runtime use
        
        config = DINOV2_CONFIGS.get(model_name, DINOV2_CONFIGS["dinov2_vits14"])
        embed_dim = config["embed_dim"]
        patch_size = config["patch_size"]
        
        print(f"Initializing {model_name} with embed_dim={embed_dim}")
        if gate_config and gate_config.enabled:
            print(f"Gated attention enabled: headwise={gate_config.headwise}, elementwise={gate_config.elementwise}")
        
        # Load pretrained backbone (with optional gated attention)
        backbone = load_dinov2_backbone(model_name, checkpoint_path, gate_config)
        
        # Create teacher (frozen copy) and student
        self.teacher_backbone = DINOv2Wrapper(copy.deepcopy(backbone), patch_size)
        self.student_backbone = DINOv2Wrapper(backbone, patch_size)
        freeze_eval_module(self.teacher_backbone)

        # Projection heads
        dino_head = partial(DINOv2ProjectionHead, input_dim=embed_dim)
        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(DINOv2ProjectionHead, input_dim=embed_dim)
        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head,
            ibot_head=teacher_ibot_head,
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head,
            ibot_head=student_ibot_head,
        )
        freeze_eval_module(self.teacher_head)

        # Losses
        self.dino_criterion = DINOLoss()
        self.ibot_criterion = IBOTPatchLoss()
        self.koleo_criterion = KoLeoLoss()
        self.base_lr = base_lr
        self.freeze_backbone_epochs = freeze_backbone_epochs

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self,
        x: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features

    def training_step(
        self,
        batch: tuple[list[Tensor], Tensor, list[str]],
        batch_idx: int,
    ) -> Tensor:
        views = batch[0]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        batch_size = len(global_views)
        sequence_length = self.teacher_backbone.sequence_length
        mask = global_views.new_zeros((batch_size, sequence_length), dtype=torch.bool)

        H, W = self.teacher_backbone.grid_size
        assert (
            H * W == sequence_length - 1
        ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
        block_mask = random_block_mask(size=(batch_size, H, W), device=mask.device)
        mask[:, 1:] = block_mask.flatten(start_dim=1)

        with torch.no_grad():
            teacher_cls_token, teacher_features = self.forward_teacher(global_views)
            teacher_cls_out = self.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = self.teacher_head.ibot_head.forward(
                teacher_features[mask]
            )

        student_global_cls_token, student_global_masked_features = self.forward_student(
            global_views,
            mask=mask,
        )
        student_global_cls_out = self.student_head.dino_head.forward(
            student_global_cls_token
        )
        student_global_masked_out = self.student_head.ibot_head.forward(
            student_global_masked_features
        )

        student_local_cls_token, _ = self.forward_student(local_views, mask=None)
        student_local_cls_out = self.student_head.dino_head.forward(
            student_local_cls_token
        )
        student_cls_out = torch.cat(
            [
                student_global_cls_out,
                student_local_cls_out,
            ]
        )

        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=int(
                30
                / max(1, self.trainer.max_epochs)
                * self.trainer.estimated_stepping_batches
            ),
            start_value=0.04,
            end_value=0.07,
        )
        dino_loss = self.dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        koleo_loss = 0.1 * sum(
            self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )
        loss = dino_loss + ibot_loss + koleo_loss
        
        # Logging
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/dino_loss", dino_loss, sync_dist=True)
        self.log("train/ibot_loss", ibot_loss, sync_dist=True)
        self.log("train/koleo_loss", koleo_loss, sync_dist=True)
        self.log("train/teacher_temp", teacher_temp, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        # Use lower LR for finetuning backbone vs. heads
        backbone_params = list(self.student_backbone.parameters())
        head_params = list(self.student_head.parameters())
        
        param_groups = [
            {"params": backbone_params, "lr": self.base_lr * 0.1, "name": "backbone"},  # 10x lower LR for backbone
            {"params": head_params, "lr": self.base_lr, "name": "heads"},
        ]
        optimizer = AdamW(param_groups, weight_decay=0.04)
        return optimizer

    def on_train_epoch_start(self):
        # Optionally freeze backbone for first N epochs
        if self.current_epoch < self.freeze_backbone_epochs:
            for param in self.student_backbone.parameters():
                param.requires_grad = False
            print(f"Epoch {self.current_epoch}: Backbone frozen")
        elif self.current_epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
            for param in self.student_backbone.parameters():
                param.requires_grad = True
            print(f"Epoch {self.current_epoch}: Backbone unfrozen")

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        if self.current_epoch < 1:
            for param_group in optimizer.param_groups:
                if "last_layer" in param_group.get("name", ""):
                    param_group["lr"] = 0.0

        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.04,
            end_value=0.4,
        )
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                group["weight_decay"] = weight_decay

    def on_train_batch_end(self, outputs, batch, batch_idx):
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.996,  # Higher momentum for finetuning
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        
        self.log("train/ema_momentum", momentum, sync_dist=True)
        return super().on_train_batch_end(outputs, batch, batch_idx)


def parse_args() -> Config:
    """Parse command line arguments. Supports YAML config file with CLI overrides."""
    parser = argparse.ArgumentParser(
        description="DINOv2 SSL Finetuning with Lightly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file:
  python lightly_train_dinov2.py --config configs/dinov2_finetune.yaml
  
  # Using CLI args:
  python lightly_train_dinov2.py --data-dir /path/to/train --model-name dinov2_vitg14
  
  # Config file with CLI overrides:
  python lightly_train_dinov2.py --config configs/dinov2_finetune.yaml --epochs 100
        """
    )
    
    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    
    # CLI overrides (can override config values)
    parser.add_argument("--data-dir", type=str, default=None, help="Training data directory.")
    parser.add_argument("--val-dir", type=str, default=None, help="Validation data directory for KNN eval.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint path.")
    parser.add_argument("--model-name", type=str, default=None, choices=list(DINOV2_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None, help="Precision: 32, 16, bf16-mixed")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true", default=False)
    parser.add_argument("--no-wandb", action="store_true", default=False)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--knn-t", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    # Gate config overrides
    parser.add_argument("--gate-enabled", action="store_true", default=None, help="Enable gated attention")
    parser.add_argument("--no-gate", action="store_true", default=False, help="Disable gated attention")
    parser.add_argument("--gate-headwise", action="store_true", default=None, help="Use headwise gating")
    parser.add_argument("--gate-elementwise", action="store_true", default=None, help="Use elementwise gating")
    
    args = parser.parse_args()
    
    # Load config from file or use defaults
    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config(args.config)
    else:
        config = Config()
    
    # Apply CLI overrides
    if args.data_dir is not None:
        config.data.train_dir = args.data_dir
    if args.val_dir is not None:
        config.data.val_dir = args.val_dir
        config.knn.enabled = True
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.model_name is not None:
        config.model.name = args.model_name
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.accelerator is not None:
        config.distributed.accelerator = args.accelerator
    if args.devices is not None:
        config.distributed.devices = args.devices
    if args.strategy is not None:
        config.distributed.strategy = args.strategy
    if args.precision is not None:
        config.distributed.precision = args.precision
    if args.freeze_backbone_epochs is not None:
        config.training.freeze_backbone_epochs = args.freeze_backbone_epochs
    if args.checkpoint_dir is not None:
        config.checkpoint.dir = args.checkpoint_dir
    if args.save_every_n_epochs is not None:
        config.checkpoint.save_every_n_epochs = args.save_every_n_epochs
    if args.wandb_project is not None:
        config.logging.wandb.project = args.wandb_project
    if args.wandb_name is not None:
        config.logging.wandb.name = args.wandb_name
    if args.wandb_offline:
        config.logging.wandb.offline = True
    if args.no_wandb:
        config.logging.wandb.enabled = False
    if args.knn_k is not None:
        config.knn.k = args.knn_k
    if args.knn_t is not None:
        config.knn.t = args.knn_t
    if args.seed is not None:
        config.experiment.seed = args.seed
    
    # Gate config overrides
    if args.no_gate:
        config.gate.enabled = False
    elif args.gate_enabled:
        config.gate.enabled = True
    if args.gate_headwise:
        config.gate.headwise = True
        config.gate.elementwise = False
    if args.gate_elementwise:
        config.gate.elementwise = True
        config.gate.headwise = False
    
    return config


def create_train_dataloader(config: Config) -> torch.utils.data.DataLoader:
    """Create training dataloader with DINO augmentations."""
    # DINOv2 uses patch_size=14, so crop sizes must be divisible by 14
    transform = DINOTransform(
        global_crop_size=config.data.global_crop_size,
        global_crop_scale=config.data.global_crop_scale,
        local_crop_size=config.data.local_crop_size,
        local_crop_scale=config.data.local_crop_scale,
        n_local_views=config.data.n_local_views,
        cj_hue=0.0,  # Disable hue jitter to avoid torchvision/numpy overflow bug
    )
    dataset = LightlyDataset(input_dir=config.data.train_dir, transform=transform)
    print(f"Training dataset size: {len(dataset)} images")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )


def create_val_dataloader(val_dir: str, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    """Create validation dataloader with simple transforms for KNN eval."""
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = LightlyDataset(input_dir=val_dir, transform=val_transform)
    print(f"Validation dataset size: {len(dataset)} images")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )


class KNNEvalCallback(pl.Callback):
    """Callback to run KNN evaluation at the end of each epoch."""
    
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        knn_k: int = 20,
        knn_t: float = 0.07,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.num_classes = num_classes
    
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: DINOv2Lightning):
        """Run KNN evaluation using teacher backbone."""
        device = pl_module.device
        pl_module.eval()
        
        # Extract features from training set (subsample for speed)
        train_features = []
        train_labels = []
        max_train_samples = 50000  # Limit for speed
        
        for batch in self.train_dataloader:
            if len(train_features) * batch[0].shape[0] >= max_train_samples:
                break
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Get CLS token from teacher
            features = pl_module.teacher_backbone.backbone(images)
            features = F.normalize(features, dim=1)
            train_features.append(features)
            train_labels.append(labels)
        
        train_features = torch.cat(train_features, dim=0).t().contiguous()  # (dim, N)
        train_labels = torch.cat(train_labels, dim=0)
        
        # Evaluate on validation set
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        for batch in self.val_dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            features = pl_module.teacher_backbone.backbone(images)
            features = F.normalize(features, dim=1)
            
            # KNN prediction
            pred_labels = knn_predict(
                feature=features,
                feature_bank=train_features,
                feature_labels=train_labels,
                num_classes=self.num_classes,
                knn_k=self.knn_k,
                knn_t=self.knn_t,
            )
            
            # Top-1 accuracy
            correct_top1 += (pred_labels[:, 0] == labels).sum().item()
            # Top-5 accuracy
            correct_top5 += (pred_labels[:, :5] == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
        
        top1_acc = 100 * correct_top1 / total
        top5_acc = 100 * correct_top5 / total
        
        pl_module.log("val/knn_top1", top1_acc, sync_dist=True)
        pl_module.log("val/knn_top5", top5_acc, sync_dist=True)
        print(f"KNN Eval - Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
        
        pl_module.train()


def main():
    config = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(config.experiment.seed)
    
    # Print config summary
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Model: {config.model.name}")
    print(f"  Checkpoint: {config.model.checkpoint}")
    print(f"  Gated Attention: {config.gate.enabled}")
    if config.gate.enabled:
        print(f"    - Headwise: {config.gate.headwise}")
        print(f"    - Elementwise: {config.gate.elementwise}")
    print(f"  Train data: {config.data.train_dir}")
    print(f"  Val data: {config.data.val_dir}")
    print(f"  Batch size (per GPU): {config.data.batch_size}")
    # Calculate effective batch size
    num_devices = config.distributed.devices
    if isinstance(num_devices, str):
        if num_devices == "auto":
            import torch
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            num_devices = int(num_devices)
    effective_batch = config.data.batch_size * num_devices
    print(f"  Effective batch size: {effective_batch} ({config.data.batch_size} x {num_devices} GPUs)")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Devices: {config.distributed.devices}")
    print(f"  Precision: {config.distributed.precision}")
    print(f"  Strategy: {config.distributed.strategy}")
    print(f"  KNN eval: {config.knn.enabled}")
    print("=" * 60)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint.dir, exist_ok=True)
    
    # Setup wandb logger
    logger = None
    if config.logging.wandb.enabled:
        logger = WandbLogger(
            project=config.logging.wandb.project,
            name=config.logging.wandb.name or config.experiment.name,
            offline=config.logging.wandb.offline,
            log_model=False,
        )
        logger.log_hyperparams(config_to_flat_dict(config))
    
    # Initialize model
    model = DINOv2Lightning(
        model_name=config.model.name,
        checkpoint_path=config.model.checkpoint,
        base_lr=config.training.learning_rate,
        freeze_backbone_epochs=config.training.freeze_backbone_epochs,
        gate_config=config.gate,
    )
    
    # Create dataloaders
    train_dataloader = create_train_dataloader(config)
    
    # Setup callbacks
    # Use custom filename if provided, otherwise default (avoid / in filename)
    ckpt_filename = config.checkpoint.filename or f"{config.model.name}_epoch{{epoch:02d}}"
    callbacks = [
        # Save checkpoint every N epochs
        ModelCheckpoint(
            dirpath=config.checkpoint.dir,
            filename=ckpt_filename,
            save_top_k=config.checkpoint.save_top_k,
            every_n_epochs=config.checkpoint.save_every_n_epochs,
            save_last=config.checkpoint.save_last,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Optional: KNN evaluation callback
    val_dataloader = None
    if config.knn.enabled and config.data.val_dir:
        print(f"Enabling KNN evaluation with validation data from {config.data.val_dir}")
        val_dataloader = create_val_dataloader(
            config.data.val_dir, 
            batch_size=config.data.batch_size * 2,  # Can use larger batch for eval
            num_workers=config.data.num_workers
        )
        # Need a separate dataloader for train features (without augmentation)
        train_features_dataloader = create_val_dataloader(
            config.data.train_dir,
            batch_size=config.data.batch_size * 2,
            num_workers=config.data.num_workers
        )
        callbacks.append(KNNEvalCallback(
            train_dataloader=train_features_dataloader,
            val_dataloader=val_dataloader,
            knn_k=config.knn.k,
            knn_t=config.knn.t,
            num_classes=config.knn.num_classes,
        ))

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.distributed.accelerator,
        devices=config.distributed.devices,
        strategy=config.distributed.strategy,
        precision=config.distributed.precision,
        sync_batchnorm=config.distributed.sync_batchnorm,
        use_distributed_sampler=True,
        logger=logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        callbacks=callbacks,
        val_check_interval=1.0,  # Validate every epoch
    )
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    # Save final checkpoint (backbone only)
    if trainer.is_global_zero:
        final_path = os.path.join(config.checkpoint.dir, f"{config.model.name}_finetuned_final.pth")
        torch.save(model.student_backbone.backbone.state_dict(), final_path)
        print(f"Saved final finetuned backbone to {final_path}")


if __name__ == "__main__":
    main()
