import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from lightly_train_dinov2 import DINOv2Lightning
from models.gated_attention_v2 import GatedAttentionV2, GatedMemEffAttentionV2

def get_transform(patch_size=14):
    class ResizeToMultiple:
        def __init__(self, divisor):
            self.divisor = divisor
        def __call__(self, img):
            w, h = img.size
            new_w = (w // self.divisor) * self.divisor
            new_h = (h // self.divisor) * self.divisor
            return img.resize((new_w, new_h), Image.BICUBIC)

    return transforms.Compose([
        ResizeToMultiple(patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

class GateAuditor:
    def __init__(self, model):
        self.model = model
        self.data = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # We want to hook into the GatedAttentionV2 modules
        print("Registering hooks...")
        count = 0
        for name, module in self.model.named_modules():
            # Debug: Print module types to ensure we are matching correctly
            if "attn" in name and "gate" in name: 
                 # Just a loose check to see if we are close
                 pass 
            
            if hasattr(module, "gate") and module.gate is not None:
                print(f"  Hooking {name} ({type(module).__name__})")
                self.hooks.append(module.register_forward_hook(self._get_audit_hook(name, module)))
                count += 1
        
        print(f"Registered {count} hooks.")


    def _get_audit_hook(self, name, attn_module):
        def hook(module, inputs, output):
            # inputs[0] is x (B, N, C)
            x_in = inputs[0]
            B, N, C = x_in.shape
            
            # --- 1. Recompute Pre-Gate Attention Output (x_pre) ---
            # Note: We assume evaluation mode (no dropout)
            
            # QKV
            qkv = attn_module.qkv(x_in)
            qkv = qkv.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
            
            # Standardize shape to (3, B, H, N, D) for unbinding
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = q * attn_module.scale
            
            # Attention
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            # x_pre = (attn @ v).transpose(1, 2).reshape(B, N, C) 
            # Wait, we want per-head features before reshape for analysis
            x_pre_head = (attn @ v).transpose(1, 2) # (B, N, H, D)
            
            # --- 2. Get Gate Values ---
            # We need to recompute gate because we can't easily access the intermediate 'gate_score' 
            # from the module output (which is post-proj).
            gate_score = attn_module.gate(x_in)
            gate_val = torch.sigmoid(gate_score) # (B, N, gate_dim)
            
            # Reshape gate to match (B, N, H, D) or (B, N, H, 1)
            if attn_module.elementwise_gate:
                gate_val = gate_val.reshape(B, N, attn_module.num_heads, attn_module.head_dim)
            elif attn_module.headwise_gate:
                gate_val = gate_val.reshape(B, N, attn_module.num_heads, 1)
            
            # Store
            if name not in self.data:
                self.data[name] = {}
            
            # Detach and CPU to save memory
            self.data[name] = {
                "x_pre": x_pre_head.detach().cpu(), # (B, N, H, D)
                "gate": gate_val.detach().cpu()     # (B, N, H, D/1)
            }
            
        return hook
    
    def clear(self):
        self.data = {}

def audit_image(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    # Gate Config
    gate_config = None
    if 'gate_config_dict' in hparams:
        from lightly_train_dinov2 import GateConfig
        gate_config = GateConfig(**hparams['gate_config_dict'])
    
    hparams['checkpoint_path'] = None 
    model = DINOv2Lightning(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # 2. Prepare Image
    transform = get_transform()
    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 3. Forward Pass with Audit
    # Switch to STUDENT backbone as requested
    print("Auditing STUDENT backbone...")
    auditor = GateAuditor(model.student_backbone.backbone)
    
    with torch.no_grad():
        # We need to manually call student backbone because model(x) calls teacher
        # transform(img) -> unsqueeze -> to(device) -> x
        # model.student_backbone.encode(x) calls backbone.forward_features
        # We hooked backbone directly.
        # Let's call backbone(x) directly to trigger hooks
        _ = model.student_backbone.backbone(img_tensor)
        
    # 4. Analysis
    # Analyze ALL layers
    layer_names = sorted(auditor.data.keys(), key=lambda x: int(x.split('.')[-2]))
    target_layers = layer_names # All layers
    
    print(f"\nAnalyzing {len(target_layers)} layers: {[l.split('.')[-2] for l in target_layers]}")
    
    # Collect statistics
    dead_heads_stats = []
    alive_heads_stats = []
    
    for name in target_layers:
        data = auditor.data[name]
        x_pre = data["x_pre"] 
        gate = data["gate"]   
        
        gate_per_head = gate.mean(dim=(0, 1, 3)) if gate.dim() == 4 else gate.mean(dim=(0, 1, 3))
        x_norm = torch.norm(x_pre, p=2, dim=-1)
        norm_per_head = x_norm.mean(dim=(0, 1))
        
        layer_idx = int(name.split('.')[-2])
        
        for h in range(len(gate_per_head)):
            g_val = gate_per_head[h].item()
            n_val = norm_per_head[h].item()
            
            stat = {
                "layer": layer_idx,
                "head": h,
                "gate_mean": g_val,
                "pre_gate_norm_mean": n_val,
                "layer_name": name
            }
            
            # Relax threshold slightly to catch dampened heads
            if g_val < 0.1: 
                dead_heads_stats.append(stat)
            elif g_val > 0.9:
                alive_heads_stats.append(stat)

    print(f"Found {len(dead_heads_stats)} Dead Heads (Gate < 0.1)")
    print(f"Found {len(alive_heads_stats)} Alive Heads (Gate > 0.9)")
    
    # --- Comparison 1: Scatter Plot (Gate vs Norm) ---
    # Collect all heads from target layers
    all_gates = []
    all_norms = []
    for name in target_layers:
        data = auditor.data[name]
        gate = data["gate"].mean(dim=(0, 1, 3))
        norm = torch.norm(data["x_pre"], p=2, dim=-1).mean(dim=(0, 1))
        all_gates.extend(gate.numpy())
        all_norms.extend(norm.numpy())
        
    plt.figure(figsize=(8, 6))
    plt.scatter(all_norms, all_gates, alpha=0.6)
    plt.xlabel("Pre-Gate Feature Norm (Mean)")
    plt.ylabel("Gate Value (Mean)")
    plt.title(f"Gate Value vs. Feature Norm (Last 5 Layers)\n{os.path.basename(args.image)}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "audit_scatter.png"))
    plt.close()
    
    # --- Comparison 2: Visualizing "The Ghosts" ---
    # Pick Top-3 "Loudest Dead Heads" (Highest Norm among Dead)
    # Pick Top-3 "Loudest Alive Heads" (Highest Norm among Alive) - to contrast
    
    dead_heads_stats.sort(key=lambda x: x["pre_gate_norm_mean"], reverse=True)
    alive_heads_stats.sort(key=lambda x: x["pre_gate_norm_mean"], reverse=True)
    
    top_dead = dead_heads_stats[:3]
    top_alive = alive_heads_stats[:3]
    
    if not top_dead:
        print("No dead heads found! Skipping visualization.")
        return

    # Create grid: 3 columns (Dead 1, Dead 2, Dead 3)
    # Row 1: Pre-Gate Norm (Ghost)
    # Row 2: Gate Map
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # H, W for reshape
    _, _, H_img, W_img = img_tensor.shape
    h_patches = H_img // 14
    w_patches = W_img // 14
    
    for i, stat in enumerate(top_dead):
        layer_name = stat["layer_name"]
        head_idx = stat["head"]
        
        data = auditor.data[layer_name]
        
        # Get spatial map of Pre-Gate Norm
        # x_pre: (1, N, H, D) -> Select Head -> (1, N, D) -> Norm -> (N,)
        x_pre_head = data["x_pre"][:, :, head_idx, :]
        norms = torch.norm(x_pre_head, p=2, dim=-1).squeeze(0)
        
        # Reshape (skip CLS)
        patch_norms = norms[1:].reshape(h_patches, w_patches).numpy()
        
        # Get spatial map of Gate
        # gate: (1, N, H, D) -> Select Head -> Mean D -> (N,)
        gate_head = data["gate"][:, :, head_idx, :].mean(dim=-1).squeeze(0)
        patch_gates = gate_head[1:].reshape(h_patches, w_patches).numpy()
        
        # Plot Pre-Gate (Ghost)
        im1 = axes[0, i].imshow(patch_norms, cmap='viridis')
        axes[0, i].set_title(f"DEAD Head L{stat['layer']} H{head_idx}\nPre-Gate Norm: {stat['pre_gate_norm_mean']:.1f}")
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Plot Gate
        im2 = axes[1, i].imshow(patch_gates, cmap='RdYlBu', vmin=0, vmax=1)
        axes[1, i].set_title(f"Gate Value (Avg: {stat['gate_mean']:.2f})")
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        
    plt.suptitle(f"Audit of Suppressed Heads (The Ghosts)\n{os.path.basename(args.image)}")
    plt.savefig(os.path.join(args.output_dir, "audit_ghosts.png"))
    plt.close()

    # --- Comparison 3: Alive Heads ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, stat in enumerate(top_alive):
        layer_name = stat["layer_name"]
        head_idx = stat["head"]
        data = auditor.data[layer_name]
        
        x_pre_head = data["x_pre"][:, :, head_idx, :]
        norms = torch.norm(x_pre_head, p=2, dim=-1).squeeze(0)
        patch_norms = norms[1:].reshape(h_patches, w_patches).numpy()
        
        im = axes[i].imshow(patch_norms, cmap='viridis')
        axes[i].set_title(f"ALIVE Head L{stat['layer']} H{head_idx}\nNorm: {stat['pre_gate_norm_mean']:.1f}")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
    plt.suptitle(f"Reference: Active Heads (The Survivors)\n{os.path.basename(args.image)}")
    plt.savefig(os.path.join(args.output_dir, "audit_survivors.png"))
    plt.close()
    
    print(f"Audit complete. Results in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="audit_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    audit_image(args)

