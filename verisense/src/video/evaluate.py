"""
VeriSense Video Branch Evaluation Script
Computes video-level AUC, accuracy, per-second prediction profiles, and cross-dataset metrics.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import load_config
from src.common.metrics import compute_binary_metrics
from src.common.utils import get_device, load_checkpoint
from src.video.dataset import VideoEmbeddingDataset
from src.video.model import VideoTemporalDetector


@torch.no_grad()
def evaluate_video_model(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = get_device()
    print(f"[VeriSense Video Eval] Evaluating {checkpoint_path} on {device}")

    model = VideoTemporalDetector(
        input_dim=config.temporal_model.input_dim,
        d_model=config.temporal_model.d_model,
        nhead=config.temporal_model.nhead,
        num_layers=config.temporal_model.num_layers,
        dim_feedforward=config.temporal_model.dim_feedforward,
        dropout=0.0,
        model_type=config.temporal_model.type,
        max_frames=config.temporal_model.max_frames,
    ).to(device)

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    test_ds = VideoEmbeddingDataset(
        config.data.test_split_csv,
        max_frames=config.temporal_model.max_frames,
        embedding_dim=config.temporal_model.input_dim,
    )
    test_loader = DataLoader(test_ds, batch_size=config.data.val_batch_size, shuffle=False)

    all_preds, all_labels, video_ids = [], [], []

    for batch in tqdm(test_loader, desc="Testing Video"):
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        video_logits, frame_logits = model(embeddings, mask=mask)
        probs = torch.softmax(video_logits, dim=-1)[:, 1]

        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        video_ids.extend(batch["video_id"])

    metrics = compute_binary_metrics(np.array(all_labels), np.array(all_preds))

    print("\n" + "=" * 50)
    print("VIDEO BRANCH TEST METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k.upper():<20}: {v:.4f}")
        else:
            print(f"{k.upper():<20}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Branch")
    parser.add_argument("--config", type=str, default="configs/video.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/video_best.pt")
    args = parser.parse_args()

    evaluate_video_model(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
