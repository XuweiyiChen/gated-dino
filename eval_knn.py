import torch
import torch.nn as nn
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from lightly_train_dinov2 import DINOv2Lightning
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.data import LightlyDataset
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class StudentBackboneWrapper(nn.Module):
    """Wrapper to extract CLS token from student backbone for k-NN evaluation."""
    
    def __init__(self, lightning_model: DINOv2Lightning):
        super().__init__()
        self.lightning_model = lightning_model
        self.lightning_model.eval()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS token from student backbone."""
        with torch.no_grad():
            # Use student backbone - returns CLS token
            features = self.lightning_model.student_backbone.backbone(images)
            # features shape: (batch_size, embed_dim)
            return features


def evaluate_knn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint['hyper_parameters']
    
    hparams['checkpoint_path'] = None
    model = DINOv2Lightning(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    
    # Wrap model for k-NN evaluation
    backbone = StudentBackboneWrapper(model)
    
    # 2. Data Loaders
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ])
    
    print(f"Loading Train Data (Bank) from: {args.train_dir}")
    train_dataset = LightlyDataset(input_dir=str(args.train_dir), transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    
    print(f"Loading Val Data (Query) from: {args.val_dir}")
    val_dataset = LightlyDataset(input_dir=str(args.val_dir), transform=transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 3. Create KNN Classifier
    classifier = KNNClassifier(
        model=backbone,
        num_classes=args.num_classes,
        knn_k=args.k,
        knn_t=args.t,
        topk=(1, 5),
    )
    
    # 4. Run Evaluation
    print("Running k-NN evaluation...")
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=TensorBoardLogger(save_dir=str(args.log_dir), name="knn_eval"),
        callbacks=[
            DeviceStatsMonitor(),
            metric_callback,
        ],
        num_sanity_val_steps=0,
    )
    
    trainer.validate(
        model=classifier,
        dataloaders=[train_dataloader, val_dataloader],
        verbose=True,
    )
    
    # 5. Print Results
    print("\n" + "="*60)
    print(f"Results for {args.checkpoint}")
    print(f"Student Backbone | k={args.k} | t={args.t}")
    
    metrics_dict = {}
    for metric in ["val_top1", "val_top5"]:
        for name, values in metric_callback.val_metrics.items():
            if name.startswith(metric):
                max_val = max(values)
                print(f"{name}: {max_val:.2f}%")
                metrics_dict[name] = max_val
    
    print("="*60)
    
    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="data/imagenet-1k/train")
    parser.add_argument("--val_dir", type=str, default="data/imagenet-1k/val")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--t", type=float, default=0.07)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default="logs/knn_eval")
    
    args = parser.parse_args()
    evaluate_knn(args)
