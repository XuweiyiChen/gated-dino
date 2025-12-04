import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
from tqdm import tqdm

from models import replace_attention_with_gated_v2
from lightly_train_dinov2 import DINOv2Lightning

# Use the same transform logic as inference to ensure consistency
def get_transform(patch_size=14):
    # We need to ensure image dimensions are multiples of patch_size
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

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    # Handle GateConfig reconstruction
    gate_config = None
    if 'gate_config_dict' in hparams:
        from lightly_train_dinov2 import GateConfig
        gate_config = GateConfig(**hparams['gate_config_dict'])
    
    # Initialize model
    # Set checkpoint_path to None in hparams to avoid reloading from disk inside init
    hparams['checkpoint_path'] = None 
    
    model = DINOv2Lightning(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # We need to access the underlying DINOv2 model
    backbone = model.student_backbone.backbone
    
    return backbone, gate_config

class GateAnalyzer:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Find all GatedAttentionV2 modules
        for name, module in self.model.named_modules():
            if "attn" in name and hasattr(module, "gate") and module.gate is not None:
                # Hook onto the gate linear layer to get raw logits
                # We also want the input to the attention block to calculate norms
                
                # Hook for Input Norms (Pre-Gate)
                # The input to GatedAttentionV2.forward is 'x'
                self.hooks.append(module.register_forward_pre_hook(
                    self._get_input_hook(name)
                ))
                
                # Hook for Gate Values (Output of Gate Linear)
                self.hooks.append(module.gate.register_forward_hook(
                    self._get_gate_output_hook(name, module)
                ))

                # Hook for Output Norms (Post-Gate & Post-Proj)
                self.hooks.append(module.register_forward_hook(
                    self._get_output_hook(name)
                ))

    def _get_output_hook(self, name):
        def hook(module, inputs, output):
            # output is (B, N, C)
            norms = torch.norm(output, p=2, dim=-1).detach().cpu()
            if name not in self.activations:
                self.activations[name] = {}
            self.activations[name]["output_norms"] = norms
        return hook


    def _get_input_hook(self, name):
        def hook(module, args):
            x = args[0] # (B, N, C)
            # Calculate per-token L2 norm
            norms = torch.norm(x, p=2, dim=-1).detach().cpu()
            if name not in self.activations:
                self.activations[name] = {}
            self.activations[name]["input_norms"] = norms
        return hook

    def _get_gate_output_hook(self, name, attn_module):
        def hook(layer, inputs, output):
            # inputs[0] is input x (B, N, C)
            x_in = inputs[0]
            
            # --- 1. Recompute Attention Output (Pre-Gate) ---
            # We need to replicate the logic inside GatedAttentionV2.forward up to gating
            # attn_module has: qkv, scale, attn_drop, proj... wait.
            # The gate is applied AFTER attention and BEFORE projection?
            # Let's check GatedAttentionV2.forward:
            # x = attn @ v
            # x = x.transpose(1, 2)
            # if gate_score is not None: x = x * sigmoid(gate_score)
            # x = x.reshape(B, N, C)
            # x = self.proj(x)
            
            # So we need 'x' just before sigmoid.
            
            # Get QKV
            B, N, C = x_in.shape
            qkv = attn_module.qkv(x_in)
            qkv = qkv.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
            
            # We need to handle the permutation/unbind
            # Standard: qkv.permute(2, 0, 3, 1, 4) -> (3, B, H, N, D)
            # MemEff: keep (B, N, 3, H, D)
            
            # We'll do standard implementation for analysis (slower but standard pytorch)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = q * attn_module.scale
            
            # Attention
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            # We typically ignore dropout for analysis/inference
            x_attn = (attn @ v).transpose(1, 2) # (B, N, H, D)
            
            # Calculate Norm of Attention Output (Pre-Gate)
            # This represents the "loudness" of each head's output for each token
            # (B, N, H, D) -> Norm over D -> (B, N, H)
            attn_out_norm = torch.norm(x_attn, p=2, dim=-1).detach().cpu()
            
            # --- 2. Get Gate Values ---
            # output is raw logits from gate linear layer
            gate_vals = torch.sigmoid(output).detach().cpu()
            
            # Reshape based on gate type
            # B, N, _ = gate_vals.shape
            # NOTE: We are inside a hook, 'attn_module' is the GatedAttentionV2 instance captured from closure
            if attn_module.elementwise_gate:
                gate_vals = gate_vals.reshape(B, N, attn_module.num_heads, attn_module.head_dim)
            elif attn_module.headwise_gate:
                gate_vals = gate_vals.reshape(B, N, attn_module.num_heads, 1)
                
            if name not in self.activations:
                self.activations[name] = {}
            self.activations[name]["gate_values"] = gate_vals
            self.activations[name]["attn_out_norm"] = attn_out_norm
        return hook

    def clear_activations(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def analyze_images(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    model, gate_config = load_model(args.checkpoint, device)
    analyzer = GateAnalyzer(model)
    
    # 2. Prepare Images
    image_files = []
    for pattern in args.images:
        image_files.extend(glob.glob(pattern))
    image_files = sorted(list(set(image_files)))
    print(f"Found {len(image_files)} images.")

    transform = get_transform()

    # 3. Process each image
    for img_path in tqdm(image_files):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        original_img = Image.open(img_path).convert("RGB")
        
        # Transform
        img_tensor = transform(original_img).unsqueeze(0).to(device)
        
        # Calculate patch dimensions for reshaping
        # DINOv2 patch size is 14
        patch_size = 14
        _, _, H_img, W_img = img_tensor.shape
        h_patches = H_img // patch_size
        w_patches = W_img // patch_size
        
        # Forward pass
        analyzer.clear_activations()
        with torch.no_grad():
            _ = model(img_tensor)
        
        # --- Analysis & Visualization ---
        
        # We'll aggregate data across layers for this image
        layer_names = sorted([k for k in analyzer.activations.keys()], key=lambda x: int(x.split('.')[-2])) # Sort by layer index
        
        # Data containers
        all_layer_gate_means = [] # (L, Heads)
        
        # Create a subplot for Spatial Gate Maps
        # We will plot the average gate value (averaged across heads and dim) for each layer
        # To save space, maybe just plot specific layers: Early, Middle, Late
        selected_layers_idx = [0, len(layer_names)//2, len(layer_names)-1]
        selected_layer_names = [layer_names[i] for i in selected_layers_idx]
        
        fig_spatial, axes_spatial = plt.subplots(1, len(selected_layers_idx) + 1, figsize=(20, 5))
        
        # Plot original image
        axes_spatial[0].imshow(original_img)
        axes_spatial[0].set_title("Original")
        axes_spatial[0].axis('off')

        for idx, layer_name in enumerate(selected_layer_names):
            data = analyzer.activations[layer_name]
            gate_vals = data["gate_values"] # (1, N, H, D) or (1, N, H, 1)
            
            # Average across heads and dimension to get a single scalar per token
            # (1, N, H, D) -> (1, N)
            spatial_gate = gate_vals.mean(dim=(2, 3)).squeeze(0) 
            
            # Remove CLS and Register tokens if any
            # DINOv2 usually: CLS + Patches. (N = 1 + h*w + registers)
            # We assume standard DINOv2: N = 1 + h*w
            # If registers are present (N > 1 + h*w), we need to handle that.
            # Check expected patches
            expected_patches = h_patches * w_patches
            if spatial_gate.shape[0] == expected_patches + 1:
                # Has CLS
                patch_gates = spatial_gate[1:]
            elif spatial_gate.shape[0] == expected_patches + 5:
                # Has CLS + 4 Registers (just in case)
                patch_gates = spatial_gate[5:]
            else:
                # Assume just CLS at 0
                 patch_gates = spatial_gate[1:]
            
            # Reshape to 2D
            patch_gates_2d = patch_gates.reshape(h_patches, w_patches).numpy()
            
            # Plot
            im = axes_spatial[idx+1].imshow(patch_gates_2d, cmap='RdBu_r', vmin=0, vmax=1) # Red=Low(Closed), Blue=High(Open) ? RdBu_r: Blue=High, Red=Low
            # Actually let's use simple viridis or plasma. Or consistent with gate meaning:
            # 1.0 (Open) -> Light/White/Blue
            # 0.0 (Closed) -> Dark/Red/Black
            # Let's use 'coolwarm_r': Blue=0 (Closed), Red=1 (Open). Wait.
            # Usually 'jet' or 'viridis'.
            # Let's use 'RdYlBu': Red=Low (Closed), Blue=High (Open).
            
            axes_spatial[idx+1].imshow(patch_gates_2d, cmap='RdYlBu', vmin=0, vmax=1)
            axes_spatial[idx+1].set_title(f"Layer {layer_name.split('.')[-2]}\nAvg Gate Val")
            axes_spatial[idx+1].axis('off')
            
        plt.colorbar(im, ax=axes_spatial.ravel().tolist())
        plt.suptitle(f"Spatial Gate Distribution: {img_name}")
        plt.savefig(os.path.join(args.output_dir, f"{img_name}_spatial_gates.png"))
        plt.close()

        # --- 2. Correlation Plot: Attn Output Norm vs Gate Value ---
        # We'll aggregate all tokens from all layers for a global scatter plot for this image
        all_norms = []
        all_gates = []
        
        for layer_name in layer_names:
            data = analyzer.activations[layer_name]
            # attn_out_norm: (B=1, N, H)
            # gate_values: (B=1, N, H, D) or (B=1, N, H, 1)
            
            norms = data["attn_out_norm"].squeeze(0) # (N, H)
            
            # Average gate values over dimension D if elementwise
            gates = data["gate_values"].squeeze(0) # (N, H, D) or (N, H, 1)
            gates = gates.mean(dim=-1) # (N, H)
            
            # Flatten
            all_norms.append(norms.flatten())
            all_gates.append(gates.flatten())
            
        all_norms = torch.cat(all_norms).numpy()
        all_gates = torch.cat(all_gates).numpy()
        
        # Calculate Statistics
        corr_coef = np.corrcoef(all_norms, all_gates)[0, 1]
        mean_gate = np.mean(all_gates)
        std_gate = np.std(all_gates)
        
        print(f"\nImage: {img_name}")
        print(f"  Correlation (AttnNorm vs Gate): {corr_coef:.4f}")
        print(f"  Mean Gate Value: {mean_gate:.4f}")
        print(f"  Std Gate Value: {std_gate:.4f}")
        
        plt.figure(figsize=(10, 6))
        # Sample points if too many
        if len(all_norms) > 50000:
            indices = np.random.choice(len(all_norms), 50000, replace=False)
            plt_norms = all_norms[indices]
            plt_gates = all_gates[indices]
        else:
            plt_norms = all_norms
            plt_gates = all_gates
            
        sns.scatterplot(x=plt_norms, y=plt_gates, s=5, alpha=0.3)
        plt.xlabel("Attention Output Norm (Pre-Gate)")
        plt.ylabel("Average Gate Value")
        plt.title(f"Gate Value vs Attn Output Norm ({img_name})\nCorrelation: {corr_coef:.4f}")
        plt.ylim(-0.05, 1.05)
        plt.savefig(os.path.join(args.output_dir, f"{img_name}_correlation.png"))
        plt.close()
        
        # --- 3. Layer/Head Heatmap ---
        # Average gate value per head, per layer
        layer_head_map = []
        for layer_name in layer_names:
            data = analyzer.activations[layer_name]
            # gate_values: (1, N, H, D)
            # Average over Batch, Tokens, Dim -> (H,)
            avg_per_head = data["gate_values"].mean(dim=(0, 1, 3)).numpy()
            layer_head_map.append(avg_per_head)
            
        layer_head_map = np.stack(layer_head_map) # (L, H)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(layer_head_map, annot=False, cmap="RdYlBu", vmin=0, vmax=1)
        plt.xlabel("Head Index")
        plt.ylabel("Layer Index")
        plt.title(f"Average Gate Value per Head/Layer ({img_name})")
        plt.savefig(os.path.join(args.output_dir, f"{img_name}_head_heatmap.png"))
        plt.close()

        # --- 4. Spatial Feature Norm Heatmaps (Attention Sink Check) ---
        # We visualize the norm of the attention output (Post-Gate)
        fig_norm, axes_norm = plt.subplots(1, len(selected_layers_idx) + 1, figsize=(20, 5))
        
        axes_norm[0].imshow(original_img)
        axes_norm[0].set_title("Original")
        axes_norm[0].axis('off')

        for idx, layer_name in enumerate(selected_layer_names):
            data = analyzer.activations[layer_name]
            norms = data["output_norms"].squeeze(0) # (N,)
            
            # Remove CLS/Registers
            if norms.shape[0] == expected_patches + 1:
                patch_norms = norms[1:]
            elif norms.shape[0] == expected_patches + 5:
                patch_norms = norms[5:]
            else:
                patch_norms = norms[1:]
            
            patch_norms_2d = patch_norms.reshape(h_patches, w_patches).numpy()
            
            # Plot with consistent scale if possible, or dynamic
            im_norm = axes_norm[idx+1].imshow(patch_norms_2d, cmap='viridis')
            axes_norm[idx+1].set_title(f"Layer {layer_name.split('.')[-2]}\nFeature Norm")
            axes_norm[idx+1].axis('off')
            plt.colorbar(im_norm, ax=axes_norm[idx+1], fraction=0.046, pad=0.04)
            
        plt.suptitle(f"Spatial Feature Norms: {img_name}")
        plt.savefig(os.path.join(args.output_dir, f"{img_name}_feature_norms.png"))
        plt.close()

    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--images", type=str, nargs="+", required=True, help="Image paths or globs")
    parser.add_argument("--output_dir", type=str, default="gate_analysis", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    analyze_images(args)

