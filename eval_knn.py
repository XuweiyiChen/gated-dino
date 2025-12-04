import os
import torch
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from lightly_train_dinov2 import DINOv2Lightning, DINOV2_CONFIGS, Config, load_config, create_val_dataloader
from lightly.utils.benchmarking import knn_predict
from torchvision import transforms
from lightly.data import LightlyDataset
import numpy as np

def evaluate_knn(args):
    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    # We need to load the config from the checkpoint or a file
    # To be safe, we load the checkpoint, get hparams, and init the model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    # Handle GateConfig reconstruction
    gate_config = None
    if 'gate_config_dict' in hparams:
        from lightly_train_dinov2 import GateConfig
        gate_config = GateConfig(**hparams['gate_config_dict'])
    
    # Override checkpoint_path to avoid reloading weights inside init
    hparams['checkpoint_path'] = None
    
    model = DINOv2Lightning(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 2. Data Loaders
    # Transform: Resize(256), CenterCrop(224), Normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Loading Train Data (Bank) from: {args.train_dir}")
    # Use ImageFolder to ensure consistent class-to-idx mapping
    from torchvision.datasets import ImageFolder
    
    train_dataset = ImageFolder(root=args.train_dir, transform=transform)
    print(f"Train classes: {len(train_dataset.classes)}")
    
    print(f"Loading Val Data (Query) from: {args.val_dir}")
    val_dataset = ImageFolder(root=args.val_dir, transform=transform)
    print(f"Val classes: {len(val_dataset.classes)}")
    
    # Check class consistency
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("WARNING: Class mapping mismatch between Train and Val!")
        # We must enforce the train mapping on val if they differ
        # But ImageFolder sorts by folder name, so if folders match, we are good.
        # If Val is flat (no folders), ImageFolder fails.
    
    # Print first few samples to debug
    print(f"Train Sample 0: {train_dataset.imgs[0]}")
    print(f"Val Sample 0: {val_dataset.imgs[0]}")

    # Limit bank size for speed/memory
    if args.max_train_samples < len(train_dataset):
        print(f"Subsampling train set to {args.max_train_samples} samples")
        # Use random subset but keep targets valid
        indices = np.random.choice(len(train_dataset), args.max_train_samples, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    # 3. Extract Features
    print("Extracting feature bank...")
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for i, (images, targets, fnames) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass (Teacher backbone usually)
            # model.teacher_backbone.backbone(x)
            features = model.teacher_backbone.backbone(images)
            features = F.normalize(features, dim=1)
            
            train_features.append(features.cpu())
            train_labels.append(targets.cpu())
            
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(train_loader)} batches")
                
    train_features = torch.cat(train_features, dim=0).t() # (Dim, N)
    train_labels = torch.cat(train_labels, dim=0)
    
    print(f"Feature bank shape: {train_features.shape}")
    
    # Move bank to device for k-NN
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    
    # 4. Evaluate
    print("Evaluating on validation set...")
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, targets, fnames) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            features = model.teacher_backbone.backbone(images)
            features = F.normalize(features, dim=1)
            
            # KNN
            pred_labels = knn_predict(
                feature=features,
                feature_bank=train_features,
                feature_labels=train_labels,
                num_classes=args.num_classes,
                knn_k=args.k,
                knn_t=args.t
            )
            
            # Accuracy
            correct_top1 += (pred_labels[:, 0] == targets).sum().item()
            correct_top5 += (pred_labels[:, :5] == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)
            
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1}/{len(val_loader)} batches - Top1: {100*correct_top1/total:.2f}%")

    top1 = 100 * correct_top1 / total
    top5 = 100 * correct_top5 / total
    
    print("\n" + "="*40)
    print(f"Results for {args.checkpoint}")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print(f"Top-5 Accuracy: {top5:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="data/imagenet-1k/train")
    parser.add_argument("--val_dir", type=str, default="data/imagenet-1k/val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=50000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--t", type=float, default=0.07)
    parser.add_argument("--num_classes", type=int, default=1000)
    
    args = parser.parse_args()
    evaluate_knn(args)

