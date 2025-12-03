# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified to add gated attention output following Qwen3's approach.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Gated Attention V2 for DINOv2.

This version separates the QKV projection and the Gate projection into two
distinct linear layers. This allows for freezing the pretrained QKV weights
while training only the Gate weights (PEFT approach).

Key idea:
- qkv layer: dim -> 3*dim (Frozen pretrained weights)
- gate layer: dim -> gate_dim (Trainable new weights)
"""

import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Check for xformers availability
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class GatedAttentionV2(nn.Module):
    """
    DINOv2 Attention with optional gating on attention output (Split Architecture).
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        proj_bias: Whether to use bias in output projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
        headwise_gate: If True, add one gate scalar per head
        elementwise_gate: If True, add one gate value per head_dim
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.headwise_gate = headwise_gate
        self.elementwise_gate = elementwise_gate
        
        # Determine gate dimension
        if self.elementwise_gate:
            gate_dim = num_heads * self.head_dim
        elif self.headwise_gate:
            gate_dim = num_heads
        else:
            gate_dim = 0
        self.gate_dim = gate_dim
        
        # Split QKV and Gate projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if self.gate_dim > 0:
            self.gate = nn.Linear(dim, gate_dim, bias=qkv_bias)
        else:
            self.gate = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        B, N, C = x.shape
        
        # Compute QKV (Standard)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        
        # Compute Gate (Separate)
        gate_score = None
        if self.gate is not None:
            gate_score = self.gate(x)
            if self.elementwise_gate:
                gate_score = gate_score.reshape(B, N, self.num_heads, self.head_dim)
            elif self.headwise_gate:
                gate_score = gate_score.reshape(B, N, self.num_heads, 1)
        
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        x = x.transpose(1, 2)
        
        if gate_score is not None:
            x = x * torch.sigmoid(gate_score)
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class GatedMemEffAttentionV2(GatedAttentionV2):
    """Memory-efficient gated attention V2 using xformers or PyTorch SDPA."""

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        B, N, C = x.shape
        
        # Compute QKV
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Compute Gate
        gate_score = None
        if self.gate is not None:
            gate_score = self.gate(x)
            if self.elementwise_gate:
                gate_score = gate_score.reshape(B, N, self.num_heads, self.head_dim)
            elif self.headwise_gate:
                gate_score = gate_score.reshape(B, N, self.num_heads, 1)

        if XFORMERS_AVAILABLE and XFORMERS_ENABLED:
            q, k, v = unbind(qkv, dim=2)
            x = memory_efficient_attention(q, k, v)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            x = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            x = x.transpose(1, 2)
        
        if gate_score is not None:
            x = x * torch.sigmoid(gate_score)
        
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


def replace_attention_with_gated_v2(
    model: nn.Module,
    headwise_gate: bool = False,
    elementwise_gate: bool = True,
    use_mem_eff: bool = True,
) -> nn.Module:
    """
    Replace all Attention modules with GatedAttentionV2.
    
    Args:
        model: DINOv2 model to modify
        headwise_gate: Use one gate per attention head
        elementwise_gate: Use one gate per element (head_dim)
        use_mem_eff: Use memory-efficient attention
    
    Returns:
        Modified model with gated attention V2
    """
    # Import both local AND Facebook's DINOv2 attention classes
    from .attention import Attention as LocalAttention, MemEffAttention as LocalMemEffAttention
    
    # Try to import Facebook's DINOv2 attention classes (from torch.hub cache)
    try:
        from dinov2.layers.attention import Attention as DINOv2Attention, MemEffAttention as DINOv2MemEffAttention
        attn_classes = (LocalAttention, LocalMemEffAttention, DINOv2Attention, DINOv2MemEffAttention)
    except ImportError:
        attn_classes = (LocalAttention, LocalMemEffAttention)
    
    attn_cls = GatedMemEffAttentionV2 if use_mem_eff else GatedAttentionV2
    
    # Collect modules to replace
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, attn_classes):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            modules_to_replace.append((name, module, parent_name, attr_name))
    
    print(f"Found {len(modules_to_replace)} attention modules to replace with V2 (Split)")
    
    # Replace
    for name, old_attn, parent_name, attr_name in modules_to_replace:
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        dim = old_attn.qkv.in_features
        num_heads = old_attn.num_heads
        qkv_bias = old_attn.qkv.bias is not None
        proj_bias = old_attn.proj.bias is not None
        
        new_attn = attn_cls(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            headwise_gate=headwise_gate,
            elementwise_gate=elementwise_gate,
        )
        
        with torch.no_grad():
            # Copy QKV weights directly
            new_attn.qkv.weight.copy_(old_attn.qkv.weight)
            if qkv_bias:
                new_attn.qkv.bias.copy_(old_attn.qkv.bias)
            
            # Initialize Gate weights (zeros + bias)
            if new_attn.gate is not None:
                nn.init.zeros_(new_attn.gate.weight)
                if qkv_bias:
                    # Initialize bias to positive value (start with open gates)
                    nn.init.constant_(new_attn.gate.bias, 4.0)
            
            new_attn.proj.weight.copy_(old_attn.proj.weight)
            if proj_bias:
                new_attn.proj.bias.copy_(old_attn.proj.bias)
        
        setattr(parent, attr_name, new_attn)
    
    return model


__all__ = [
    "GatedAttentionV2",
    "GatedMemEffAttentionV2", 
    "replace_attention_with_gated_v2",
]

