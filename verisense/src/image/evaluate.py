"""
VeriSense Image Branch Evaluation Script
Evaluates model on held-out test splits, per-generator performance breakdown,
and compression/blur robustness sweeps.
"""

import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
from src.image.dataset import DeepfakeImageDataset
from src.image.model import CLIPDeepfakeDetector


def get_robustness_transform(img_size: int = 224, jpeg_quality: int = 100, blur_radius: int = 0) -> A.Compose:
    aug_list = [A.Resize(img_size, img_size)]
    if jpeg_quality < 100:
        aug_list.append(A.ImageCompression(quality_lower=jpeg_quality, quality_upper=jpeg_quality, p=1.0))
    if blur_radius > 0:
        aug_list.append(A.GaussianBlur(blur_limit=(blur_radius, blur_radius), p=1.0))

    aug_list.extend([
        A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ToTensorV2(),
    ])
    return A.Compose(aug_list)


@torch.no_grad()
def run_evaluation(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_gens = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)[:, 1]

        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_gens.extend(batch["generator"])

    return np.array(all_labels), np.array(all_preds), all_gens


def evaluate_image_model(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = get_device()
    print(f"[VeriSense Image Eval] Evaluating checkpoint: {checkpoint_path} on {device}")

    model = CLIPDeepfakeDetector(
        model_name=config.model.name,
        freeze_backbone=True,
        head_type=config.model.head_type,
        hidden_dim=config.model.hidden_dim,
        use_lora=config.model.use_lora,
    ).to(device)

    load_checkpoint(checkpoint_path, model, device=device)

    # 1. Standard Test Evaluation
    eval_transform = get_robustness_transform(img_size=config.data.image_size)
    test_dataset = DeepfakeImageDataset(config.data.test_split_csv, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.data.val_batch_size, shuffle=False)

    y_true, y_prob, generators = run_evaluation(model, test_loader, device)
    overall_metrics = compute_binary_metrics(y_true, y_prob)

    print("\n" + "=" * 50)
    print("OVERALL TEST METRICS")
    print("=" * 50)
    for k, v in overall_metrics.items():
        if isinstance(v, float):
            print(f"{k.upper():<20}: {v:.4f}")
        else:
            print(f"{k.upper():<20}: {v}")

    # 2. Per-Generator Breakdown
    df_results = pd.DataFrame({"true": y_true, "prob": y_prob, "generator": generators})
    print("\n" + "=" * 50)
    print("PER-GENERATOR BREAKDOWN")
    print("=" * 50)
    for gen_name, group in df_results.groupby("generator"):
        if len(group) > 5 and len(np.unique(group["true"])) > 1:
            m = compute_binary_metrics(group["true"].values, group["prob"].values)
            print(f"Generator [{gen_name:<16}] N={len(group):<5} | AUC: {m['auc']:.4f} | Acc: {m['accuracy']:.4f} | AP: {m['average_precision']:.4f}")

    # 3. Robustness Sweeps (JPEG Compression & Gaussian Blur)
    print("\n" + "=" * 50)
    print("ROBUSTNESS BENCHMARK (JPEG Compression)")
    print("=" * 50)
    for q in [90, 70, 50, 30]:
        rob_trans = get_robustness_transform(config.data.image_size, jpeg_quality=q)
        rob_ds = DeepfakeImageDataset(config.data.test_split_csv, transform=rob_trans)
        rob_loader = DataLoader(rob_ds, batch_size=config.data.val_batch_size, shuffle=False)
        yt, yp, _ = run_evaluation(model, rob_loader, device)
        m = compute_binary_metrics(yt, yp)
        print(f"JPEG Quality {q:02d} -> AUC: {m['auc']:.4f} | Accuracy: {m['accuracy']:.4f} | AP: {m['average_precision']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Branch")
    parser.add_argument("--config", type=str, default="configs/image.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/image_best.pt")
    args = parser.parse_args()

    evaluate_image_model(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
