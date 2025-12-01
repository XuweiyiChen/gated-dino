"""DINOv2 layer modules - copied from facebookresearch/dinov2 for local editing."""

from .attention import Attention, MemEffAttention
from .block import Block, NestedTensorBlock
from .mlp import Mlp
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .layer_scale import LayerScale
from .drop_path import DropPath
from .gated_attention import (
    GatedAttention,
    GatedMemEffAttention,
    replace_attention_with_gated,
)

__all__ = [
    # Vanilla DINOv2 layers
    "Attention",
    "MemEffAttention",
    "Block",
    "NestedTensorBlock",
    "Mlp",
    "SwiGLUFFN",
    "SwiGLUFFNFused",
    "LayerScale",
    "DropPath",
    # Gated attention (Qwen3-style)
    "GatedAttention",
    "GatedMemEffAttention",
    "replace_attention_with_gated",
]
