"""
VeriSense Audio Branch Evaluation Script
Computes EER, minDCF, and AUC over test partitions and codec degradations.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import load_config
from src.common.metrics import compute_binary_metrics, compute_eer, compute_min_dcf
from src.common.utils import get_device, load_checkpoint
from src.audio.dataset import AudioDeepfakeDataset
from src.audio.model import Wav2Vec2DeepfakeDetector


@torch.no_grad()
def evaluate_audio_model(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = get_device()
    print(f"[VeriSense Audio Eval] Evaluating {checkpoint_path} on {device}")

    model = Wav2Vec2DeepfakeDetector(
        backbone_name=config.model.backbone_name,
        freeze_feature_extractor=True,
        unfreeze_last_n_layers=0,
        hidden_dim=config.model.hidden_dim,
        dropout=0.0,
    ).to(device)

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    test_ds = AudioDeepfakeDataset(
        config.data.test_split_csv,
        target_sr=config.audio_processing.target_sample_rate,
        duration=config.audio_processing.crop_duration_seconds,
        is_training=False,
    )
    test_loader = DataLoader(test_ds, batch_size=config.data.val_batch_size, shuffle=False)

    all_preds, all_labels = [], []

    for batch in tqdm(test_loader, desc="Testing Audio"):
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        logits = model(waveforms)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)

    eer, eer_th = compute_eer(y_true, y_prob)
    min_dcf = compute_min_dcf(y_true, y_prob)
    metrics = compute_binary_metrics(y_true, y_prob)

    print("\n" + "=" * 50)
    print("AUDIO FORENSICS TEST METRICS")
    print("=" * 50)
    print(f"EQUAL ERROR RATE (EER): {eer * 100:.2f}%")
    print(f"EER THRESHOLD         : {eer_th:.4f}")
    print(f"MIN DCF               : {min_dcf:.4f}")
    print(f"AUC-ROC               : {metrics['auc']:.4f}")
    print(f"ACCURACY              : {metrics['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Branch")
    parser.add_argument("--config", type=str, default="configs/audio.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/audio_best.pt")
    args = parser.parse_args()

    evaluate_audio_model(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
