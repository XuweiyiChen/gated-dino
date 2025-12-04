import os
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from lightly_train_dinov2 import DINOv2Lightning
from models.gated_attention_v2 import GatedAttentionV2, GatedMemEffAttentionV2
from torchvision import transforms

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
        for name, module in self.model.named_modules():
            if hasattr(module, "gate") and module.gate is not None:
                self.hooks.append(module.register_forward_hook(self._get_audit_hook(name, module)))

    def _get_audit_hook(self, name, attn_module):
        def hook(module, inputs, output):
            x_in = inputs[0]
            B, N, C = x_in.shape
            
            # Recompute Attention (Pre-Gate)
            qkv = attn_module.qkv(x_in)
            qkv = qkv.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q = q * attn_module.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            # Output before gate
            x_pre_head = (attn @ v).transpose(1, 2) # (B, N, H, D)

            # Get Gate
            gate_score = attn_module.gate(x_in)
            gate_val = torch.sigmoid(gate_score)
            if attn_module.elementwise_gate:
                gate_val = gate_val.reshape(B, N, attn_module.num_heads, attn_module.head_dim)
            elif attn_module.headwise_gate:
                gate_val = gate_val.reshape(B, N, attn_module.num_heads, 1)
            
            if name not in self.data:
                self.data[name] = {}
            
            self.data[name] = {
                "x_pre": x_pre_head.detach().cpu(),
                "gate": gate_val.detach().cpu()
            }
        return hook

def create_overlay_with_legend(bg_img, heatmap, min_val, max_val, title):
    # heatmap: 0-1 float numpy array
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Resize heatmap to match bg_img
    heatmap_colored = cv2.resize(heatmap_colored, (bg_img.shape[1], bg_img.shape[0]))
    
    # Blend
    overlay = cv2.addWeighted(bg_img, 0.6, heatmap_colored, 0.4, 0)
    
    # Add Colorbar Legend on the right side
    H, W, C = overlay.shape
    legend_w = 60
    final_img = np.ones((H, W + legend_w, C), dtype=np.uint8) * 255
    final_img[:, :W, :] = overlay
    
    # Draw gradient strip
    grad_h = H - 40
    grad_w = 20
    grad = np.linspace(255, 0, grad_h).astype(np.uint8)
    grad = np.tile(grad[:, None], (1, grad_w))
    grad_colored = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    
    final_img[20:20+grad_h, W+10:W+10+grad_w, :] = grad_colored
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_img, f"{max_val:.1f}", (W+32, 30), font, 0.5, (0,0,0), 1)
    cv2.putText(final_img, f"{min_val:.1f}", (W+32, H-25), font, 0.5, (0,0,0), 1)
    
    # Add Title on top left
    cv2.putText(final_img, title, (10, 30), font, 0.6, (255, 255, 255), 2)
    
    return final_img

def audit_overlay(args):
    # ... (Load model and image as before) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    hparams['checkpoint_path'] = None 
    model = DINOv2Lightning(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    original_pil = Image.open(args.image).convert("RGB")
    transform = get_transform()
    img_tensor = transform(original_pil).unsqueeze(0).to(device)
    
    _, _, H_tensor, W_tensor = img_tensor.shape
    original_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    original_cv2 = cv2.resize(original_cv2, (W_tensor, H_tensor))
    
    print("Auditing STUDENT backbone...")
    auditor = GateAuditor(model.student_backbone.backbone)
    with torch.no_grad():
        _ = model.student_backbone.backbone(img_tensor)
        
    all_heads = []
    for name, data in auditor.data.items():
        layer_idx = int(name.split('.')[-2])
        gate_per_head = data["gate"].mean(dim=(0, 1, 3)).numpy()
        for h in range(len(gate_per_head)):
            all_heads.append({
                "layer": layer_idx,
                "head": h,
                "gate_val": gate_per_head[h],
                "layer_name": name
            })
            
    all_heads.sort(key=lambda x: x["gate_val"])
    top_dead = all_heads[:3]
    top_alive = all_heads[-3:]

    h_patches = H_tensor // 14
    w_patches = W_tensor // 14
    
    # Process Dead Heads
    for i, head_info in enumerate(top_dead):
        layer_name = head_info["layer_name"]
        head_idx = head_info["head"]
        data = auditor.data[layer_name]
        x_pre_head = data["x_pre"][:, :, head_idx, :]
        norms = torch.norm(x_pre_head, p=2, dim=-1).squeeze(0)
        
        seq_len = norms.shape[0]
        expected = h_patches * w_patches
        if seq_len == expected + 1: patch_norms = norms[1:]
        elif seq_len == expected + 5: patch_norms = norms[5:]
        else: patch_norms = norms
            
        patch_norms_2d = patch_norms.reshape(h_patches, w_patches).numpy()
        
        # Use min/max for legend
        min_val, max_val = patch_norms_2d.min(), patch_norms_2d.max()
        heatmap_norm = (patch_norms_2d - min_val) / (max_val - min_val + 1e-8)
        
        title = f"DEAD L{head_info['layer']} H{head_idx} (G={head_info['gate_val']:.2f})"
        final_img = create_overlay_with_legend(original_cv2, heatmap_norm, min_val, max_val, title)
        
        fname = f"dead_head_{i}_L{head_info['layer']}_H{head_idx}.png"
        cv2.imwrite(os.path.join(args.output_dir, fname), final_img)
        print(f"Saved {fname}")

    # Process Alive Heads
    for i, head_info in enumerate(top_alive):
        layer_name = head_info["layer_name"]
        head_idx = head_info["head"]
        data = auditor.data[layer_name]
        x_pre_head = data["x_pre"][:, :, head_idx, :]
        norms = torch.norm(x_pre_head, p=2, dim=-1).squeeze(0)
        
        seq_len = norms.shape[0]
        expected = h_patches * w_patches
        if seq_len == expected + 1: patch_norms = norms[1:]
        elif seq_len == expected + 5: patch_norms = norms[5:]
        else: patch_norms = norms
            
        patch_norms_2d = patch_norms.reshape(h_patches, w_patches).numpy()
        
        min_val, max_val = patch_norms_2d.min(), patch_norms_2d.max()
        heatmap_norm = (patch_norms_2d - min_val) / (max_val - min_val + 1e-8)
        
        title = f"ALIVE L{head_info['layer']} H{head_idx} (G={head_info['gate_val']:.2f})"
        final_img = create_overlay_with_legend(original_cv2, heatmap_norm, min_val, max_val, title)
        
        fname = f"alive_head_{i}_L{head_info['layer']}_H{head_idx}.png"
        cv2.imwrite(os.path.join(args.output_dir, fname), final_img)
        print(f"Saved {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="audit_output_overlay")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    audit_overlay(args)

