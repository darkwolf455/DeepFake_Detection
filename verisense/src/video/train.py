"""
VeriSense Video Branch Training Script
Trains temporal model over cached frame representations.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import load_config
from src.common.metrics import compute_binary_metrics, select_balanced_threshold
from sklearn.metrics import balanced_accuracy_score
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


def train_video_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum_steps=1,
                      frame_criterion=None, annotated_clip_loss=False):
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            video_logits, frame_logits = model(embeddings, mask=mask)
            # Annotated NPZ targets are local; older NPY caches retain weak clip supervision.
            frame_labels = batch.get("frame_labels", labels[:, None].expand_as(mask)).to(device)
            visible = (labels == 0) | ((frame_labels == 1) & mask).any(dim=1)
            clip_mask = visible if annotated_clip_loss else torch.ones_like(labels, dtype=torch.bool)
            # Unobserved edits supply negative frame targets, not contradictory positive clip loss.
            loss_video = criterion(video_logits[clip_mask], labels[clip_mask]) if clip_mask.any() else video_logits.sum() * 0
            loss_frame = (frame_criterion or criterion)(frame_logits[mask], frame_labels[mask])
            loss = (loss_video + 0.25 * loss_frame) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, embeddings.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate_video_epoch(model, loader, criterion, device, return_predictions=False):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    video_ids = []

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            video_logits, _ = model(embeddings, mask=mask)
            loss = criterion(video_logits, labels)
            probs = torch.softmax(video_logits, dim=-1)[:, 1]

        loss_meter.update(loss.item(), embeddings.size(0))
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        if return_predictions:
            video_ids.extend(batch["video_id"])

    metrics = compute_binary_metrics(np.array(all_labels), np.array(all_preds))
    if return_predictions:
        return loss_meter.avg, metrics, pd.DataFrame({"video_id": video_ids, "label": all_labels,
                                                     "manipulation_score": all_preds})
    return loss_meter.avg, metrics


def train_video_model(config_path: str, debug: bool = False, warm_start=None):
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
    if train_ds.source_groups.intersection(val_ds.source_groups):
        raise ValueError("Source families cross training/validation partitions")
    if set(train_ds.df["label"]) != {0, 1} or set(val_ds.df["label"]) != {0, 1}:
        raise ValueError("Both video classes are required in train and validation")
    if config.training.get("frame_supervision") == "annotated" and not all(
            str(path).endswith(".npz") for path in [*train_ds.df["embedding_path"], *val_ds.df["embedding_path"]]):
        raise ValueError("Annotated frame training requires the NPZ caches")

    batch_size = 4 if debug else config.data.batch_size
    if config.data.get("preload_embeddings", False):
        print("Preloading the small cached sequences into RAM...", flush=True)
        train_examples, val_examples = [train_ds[i] for i in range(len(train_ds))], [val_ds[i] for i in range(len(val_ds))]
    else:
        train_examples, val_examples = train_ds, val_ds
    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_examples, batch_size=config.data.val_batch_size, shuffle=False, num_workers=num_workers)

    model = VideoTemporalDetector.from_config(config.temporal_model).to(device)
    warm_digest = None
    warm_frame_supervision = "weak"
    if warm_start:
        checkpoint = torch.load(warm_start, map_location="cpu", weights_only=True)
        if dict(checkpoint["config"]["temporal_model"]) != dict(config.temporal_model):
            raise ValueError("Warm-start temporal architecture must match the selected configuration")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        warm_digest = hashlib.sha256(Path(warm_start).read_bytes()).hexdigest()
        warm_frame_supervision = checkpoint.get("frame_supervision", "weak")

    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    epochs = 2 if debug else config.training.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    frame_weights = config.training.get("frame_class_weights")
    if config.training.get("frame_supervision") == "annotated":
        targets = torch.cat([example["frame_labels"][example["mask"]] for example in train_examples])
        counts = torch.bincount(targets, minlength=2).double()
        if (counts == 0).any():
            raise ValueError("Annotated training needs both unmodified and edited frames")
        frame_weights = (counts.sum() / (2 * counts)).tolist()
        config.training.frame_class_weights = frame_weights
        print(f"Training frame counts: unmodified={int(counts[0])}, edited={int(counts[1])}", flush=True)
        visible_labels = torch.tensor([int(example["label"]) for example in train_examples
                                       if int(example["label"]) == 0 or (example["frame_labels"][example["mask"]] == 1).any()])
        clip_counts = torch.bincount(visible_labels, minlength=2).double()
        if (clip_counts == 0).any():
            raise ValueError("Both clip classes need visible training evidence")
        clip_weights = (clip_counts.sum() / (2 * clip_counts)).tolist()
        config.training.clip_class_weights = clip_weights
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(clip_weights, dtype=torch.float32, device=device))
        print(f"Clip loss uses {int(clip_counts.sum())}/{len(train_ds)} observable training clips; validation retains all usable clips.", flush=True)
    if frame_weights is not None and (len(frame_weights) != 2 or not np.isfinite(frame_weights).all()
                                      or not all(weight > 0 for weight in frame_weights)):
        raise ValueError("Frame class weights must be positive, finite training-derived weights")
    frame_criterion = (nn.CrossEntropyLoss(weight=torch.tensor(frame_weights, dtype=torch.float32, device=device))
                       if frame_weights else criterion)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.training.use_amp))

    early_stopping = EarlyStopping(patience=config.training.early_stopping_patience, mode=config.training.mode)
    best_auc = -float("inf")
    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    selected_path = ckpt_dir / config.training.checkpoint_name
    if warm_start and selected_path.resolve() == Path(warm_start).resolve():
        raise ValueError("Use a separate checkpoint destination for adaptation")
    baseline = None
    lineage = dict(warm_start_sha256=warm_digest, frame_supervision=config.training.get("frame_supervision", "weak"))
    if warm_start:
        _, baseline = evaluate_video_epoch(model, val_loader, criterion, device)
        best_auc = baseline["auc"]
        early_stopping(best_auc)
        print("Original temporal validation:", baseline, flush=True)
        baseline_lineage = dict(lineage, frame_supervision=warm_frame_supervision,
                                attempted_frame_supervision=config.training.get("frame_supervision", "weak"))
        save_checkpoint(dict(epoch=0, model_state_dict=model.state_dict(), val_metrics=baseline,
                             config=config, **baseline_lineage), ckpt_dir, filename=config.training.checkpoint_name)

    for epoch in range(1, epochs + 1):
        tr_loss = train_video_epoch(model, train_loader, optimizer, criterion, scaler, device,
                                   config.training.gradient_accumulation_steps, frame_criterion,
                                   annotated_clip_loss=config.training.get("frame_supervision") == "annotated")
        vl_loss, vl_metrics = evaluate_video_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {vl_loss:.4f} | Val AUC: {vl_metrics['auc']:.4f} | Acc: {vl_metrics['accuracy']:.4f}", flush=True)

        if vl_metrics["auc"] > best_auc:
            best_auc = vl_metrics["auc"]
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": vl_metrics, "config": config, **lineage},
                ckpt_dir,
                filename=config.training.checkpoint_name,
            )

        if early_stopping(vl_metrics["auc"]):
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Video training finished. Best Val AUC: {best_auc:.4f}")
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    _, _, predictions = evaluate_video_epoch(model, val_loader, criterion, device, return_predictions=True)
    labels, scores = predictions["label"].to_numpy(), predictions["manipulation_score"].to_numpy()
    cutoff = select_balanced_threshold(labels, scores)
    metrics = compute_binary_metrics(labels, scores, cutoff)
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, scores >= cutoff))
    selected.update(decision_threshold=cutoff, threshold_source="validation")
    save_checkpoint(selected, ckpt_dir, filename=config.training.checkpoint_name)
    predictions.to_csv(ckpt_dir / "video_val_predictions.csv", index=False)
    report = dict(best_epoch=selected["epoch"], improved=selected["epoch"] > 0 if baseline else None,
                  baseline_validation=baseline,
                  selected_validation=metrics, decision_threshold=cutoff, threshold_source="validation",
                  scope="Visual-only validation on usable cached clips; test evaluation is pending.")
    (ckpt_dir / "video_val_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Train Video Temporal Branch")
    parser.add_argument("--config", type=str, default="configs/video.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warm-start", type=Path, help="Existing temporal checkpoint; original retained as epoch-zero candidate")
    args = parser.parse_args()

    train_video_model(args.config, args.debug, args.warm_start)


if __name__ == "__main__":
    main()
