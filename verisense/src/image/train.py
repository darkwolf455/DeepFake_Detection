"""
VeriSense Image Branch Training Script
Includes:
- AdamW optimizer & CosineAnnealingLR with warmup
- Label smoothing loss
- Class-balanced sampling
- Automatic Mixed Precision (AMP)
- Early stopping on validation AUC
- Best and last checkpoint preservation
- Fast smoke-test mode (--debug)
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import load_config
from src.common.metrics import compute_binary_metrics
from src.common.utils import (
    AverageMeter,
    EarlyStopping,
    get_device,
    save_checkpoint,
    set_seed,
    setup_logging,
)
from src.image.dataset import DeepfakeImageDataset, get_image_transforms
from src.image.model import CLIPDeepfakeDetector


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]

        loss_meter.update(loss.item(), images.size(0))
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_binary_metrics(np.array(all_labels), np.array(all_preds))
    return loss_meter.avg, metrics


def train_image_model(config_path: str, debug: bool = False):
    config = load_config(config_path)
    set_seed(config.training.seed)
    setup_logging()

    device = get_device()
    print(f"[VeriSense Image Train] Initializing on {device} (Debug={debug})")

    # Data loaders
    num_workers = 0 if platform.system() == "Windows" else config.data.num_workers
    train_transform = get_image_transforms(config, is_training=True)
    val_transform = get_image_transforms(config, is_training=False)

    limit = 64 if debug else None
    train_dataset = DeepfakeImageDataset(config.data.train_split_csv, transform=train_transform, debug_limit=limit)
    val_dataset = DeepfakeImageDataset(config.data.val_split_csv, transform=val_transform, debug_limit=limit)

    sampler = train_dataset.get_sampler() if config.data.class_balanced_sampling and not debug else None
    batch_size = 8 if debug else config.data.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.val_batch_size if not debug else 8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Instantiate model
    model = CLIPDeepfakeDetector(
        model_name=config.model.name,
        freeze_backbone=config.model.freeze_backbone,
        head_type=config.model.head_type,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        use_lora=config.model.use_lora,
    ).to(device)

    # Optimizer, Loss, Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.training.lr, weight_decay=config.training.weight_decay)
    epochs = 2 if debug else config.training.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.training.use_amp))

    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode=config.training.mode,
    )

    best_val_auc = 0.0
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Beginning training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            grad_accum_steps=config.training.gradient_accumulation_steps,
        )
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        val_auc = val_metrics["auc"]
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                checkpoint_dir,
                filename=config.training.checkpoint_name,
            )

        # Save last checkpoint
        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_metrics},
            checkpoint_dir,
            filename="image_last.pt",
        )

        if early_stopping(val_auc):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Image branch training completed! Best Validation AUC: {best_val_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train VeriSense Image Branch")
    parser.add_argument("--config", type=str, default="configs/image.yaml", help="Path to config YAML")
    parser.add_argument("--debug", action="store_true", help="Run quick 2-epoch smoke test on tiny subset")
    args = parser.parse_args()

    train_image_model(args.config, args.debug)


if __name__ == "__main__":
    main()
