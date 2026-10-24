"""
VeriSense Audio Branch Training Script
Includes:
- Weighted cross-entropy or Focal Loss
- EER (Equal Error Rate) as primary validation tracking metric
- Cosine LR scheduling with warmup
- AMP and gradient accumulation
- Debug mode for rapid testing
"""

import argparse
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import load_config
from src.common.metrics import compute_binary_metrics, compute_eer
from src.common.utils import (
    AverageMeter,
    EarlyStopping,
    get_device,
    save_checkpoint,
    set_seed,
    setup_logging,
)
from src.audio.dataset import AudioDeepfakeDataset
from src.audio.model import Wav2Vec2DeepfakeDetector


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_audio_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum_steps=1):
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, waveforms.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate_audio_epoch(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(waveforms)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]

        loss_meter.update(loss.item(), waveforms.size(0))
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    eer, threshold = compute_eer(y_true, y_prob)
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["eer"] = eer
    metrics["eer_threshold"] = threshold

    return loss_meter.avg, metrics


def train_audio_model(config_path: str, debug: bool = False):
    config = load_config(config_path)
    set_seed(config.training.seed)
    setup_logging()

    device = get_device()
    print(f"[VeriSense Audio Train] Starting on {device} (Debug={debug})")

    num_workers = 0 if platform.system() == "Windows" else config.data.num_workers
    limit = 32 if debug else None

    train_ds = AudioDeepfakeDataset(
        config.data.train_split_csv,
        target_sr=config.audio_processing.target_sample_rate,
        duration=config.audio_processing.crop_duration_seconds,
        is_training=True,
        debug_limit=limit,
    )
    val_ds = AudioDeepfakeDataset(
        config.data.val_split_csv,
        target_sr=config.audio_processing.target_sample_rate,
        duration=config.audio_processing.crop_duration_seconds,
        is_training=False,
        debug_limit=limit,
    )

    batch_size = 4 if debug else config.data.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.data.val_batch_size, shuffle=False, num_workers=num_workers)

    model = Wav2Vec2DeepfakeDetector(
        backbone_name=config.model.backbone_name,
        freeze_feature_extractor=config.model.freeze_feature_extractor,
        unfreeze_last_n_layers=config.model.unfreeze_last_n_layers,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.training.lr, weight_decay=config.training.weight_decay)
    epochs = 2 if debug else config.training.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = FocalLoss(gamma=config.training.focal_gamma) if config.training.loss_type == "focal" else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.training.use_amp))

    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode="min")
    best_eer = 1.0
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss = train_audio_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl_loss, vl_metrics = evaluate_audio_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val EER: {vl_metrics['eer']*100:.2f}% | Val AUC: {vl_metrics['auc']:.4f}")

        if vl_metrics["eer"] < best_eer:
            best_eer = vl_metrics["eer"]
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": vl_metrics, "config": config},
                ckpt_dir,
                filename=config.training.checkpoint_name,
            )

        if early_stopping(vl_metrics["eer"]):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Audio training completed. Best Val EER: {best_eer*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Audio Branch")
    parser.add_argument("--config", type=str, default="configs/audio.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    train_audio_model(args.config, args.debug)


if __name__ == "__main__":
    main()
