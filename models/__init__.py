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
from .gated_attention_v2 import (
    GatedAttentionV2,
    GatedMemEffAttentionV2,
    replace_attention_with_gated_v2,
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
    # Gated attention V1 (Qwen3-style, fused)
    "GatedAttention",
    "GatedMemEffAttention",
    "replace_attention_with_gated",
    # Gated attention V2 (split, allows freezing backbone)
    "GatedAttentionV2",
    "GatedMemEffAttentionV2",
    "replace_attention_with_gated_v2",
]
