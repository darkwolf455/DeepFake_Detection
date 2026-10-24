"""
VeriSense Video Branch Training Script
Trains temporal model over cached frame representations.
"""

import argparse
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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
from src.video.dataset import VideoEmbeddingDataset
from src.video.model import VideoTemporalDetector


def train_video_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum_steps=1):
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            video_logits, frame_logits = model(embeddings, mask=mask)
            loss_video = criterion(video_logits, labels)
            loss = loss_video / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, embeddings.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate_video_epoch(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            video_logits, _ = model(embeddings, mask=mask)
            loss = criterion(video_logits, labels)
            probs = torch.softmax(video_logits, dim=-1)[:, 1]

        loss_meter.update(loss.item(), embeddings.size(0))
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics = compute_binary_metrics(np.array(all_labels), np.array(all_preds))
    return loss_meter.avg, metrics


def train_video_model(config_path: str, debug: bool = False):
    config = load_config(config_path)
    set_seed(config.training.seed)
    setup_logging()

    device = get_device()
    print(f"[VeriSense Video Train] Initializing on {device} (Debug={debug})")

    num_workers = 0 if platform.system() == "Windows" else config.data.num_workers
    limit = 32 if debug else None

    train_ds = VideoEmbeddingDataset(
        config.data.train_split_csv,
        max_frames=config.temporal_model.max_frames,
        embedding_dim=config.temporal_model.input_dim,
        debug_limit=limit,
    )
    val_ds = VideoEmbeddingDataset(
        config.data.val_split_csv,
        max_frames=config.temporal_model.max_frames,
        embedding_dim=config.temporal_model.input_dim,
        debug_limit=limit,
    )

    batch_size = 4 if debug else config.data.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.data.val_batch_size, shuffle=False, num_workers=num_workers)

    model = VideoTemporalDetector(
        input_dim=config.temporal_model.input_dim,
        d_model=config.temporal_model.d_model,
        nhead=config.temporal_model.nhead,
        num_layers=config.temporal_model.num_layers,
        dim_feedforward=config.temporal_model.dim_feedforward,
        dropout=config.temporal_model.dropout,
        model_type=config.temporal_model.type,
        max_frames=config.temporal_model.max_frames,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    epochs = 2 if debug else config.training.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.training.use_amp))

    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode=config.training.mode)
    best_auc = 0.0
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss = train_video_epoch(model, train_loader, optimizer, criterion, scaler, device, config.training.gradient_accumulation_steps)
        vl_loss, vl_metrics = evaluate_video_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | Val AUC: {vl_metrics['auc']:.4f} | Acc: {vl_metrics['accuracy']:.4f}")

        if vl_metrics["auc"] > best_auc:
            best_auc = vl_metrics["auc"]
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": vl_metrics, "config": config},
                ckpt_dir,
                filename=config.training.checkpoint_name,
            )

        if early_stopping(vl_metrics["auc"]):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Video training finished. Best Val AUC: {best_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Video Temporal Branch")
    parser.add_argument("--config", type=str, default="configs/video.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    train_video_model(args.config, args.debug)


if __name__ == "__main__":
    main()
