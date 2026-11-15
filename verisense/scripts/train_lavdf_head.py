"""Adapt a visual/audio classifier on cached train/val embeddings; encoders stay fixed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.metrics import compute_binary_metrics, select_balanced_threshold
from src.common.utils import set_seed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_head(checkpoint):
    if checkpoint["class_names"] != ["real", "manipulated"]:
        raise ValueError("Unsupported checkpoint labels")
    hidden, dropout = checkpoint["hidden_dim"], checkpoint["dropout"]
    if checkpoint["model_name"] == "facebook/wav2vec2-base":
        if checkpoint["sample_rate"] != 16000 or checkpoint["duration"] != 4.:
            raise ValueError("Unsupported audio preprocessing")
        head = nn.Sequential(nn.Linear(768, hidden), nn.LayerNorm(hidden), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(hidden, 2))
        head.load_state_dict({key.removeprefix("head."): value for key, value in checkpoint["head_state_dict"].items()
                              if key.startswith("head.")}, strict=True)
        return head
    if checkpoint["image_size"] != 224:
        raise ValueError("Unsupported image preprocessing")
    if checkpoint["head_type"] == "mlp":
        head = nn.Sequential(nn.Linear(768, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(hidden, 2))
    elif checkpoint["head_type"] == "linear":
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 2))
    else:
        raise ValueError("Unsupported image head architecture")
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    return head


def load_cache(directory, modality="visual"):
    audio = modality == "audio"
    flag, target_key, clip_key = ("modify_audio", "window_labels", "audio_label") if audio else ("modify_video", "frame_labels", "video_label")
    unit = "window" if audio else "frame"
    tables, datasets, coverage = [], {}, {}
    for split in ("train", "val"):
        table = pd.read_csv(directory / f"{split}.csv", dtype={"group_id": str, "source_file": str})
        if (table.empty or not table["split"].eq(split).all()
                or table[["group_id", "source_file", "label", flag, "status"]].isna().any().any()
                or not table["status"].isin(["ok", "error"]).all()
                or not table["label"].eq(table[flag].astype(int)).all()):
            raise ValueError(f"Invalid {modality} labels or source assignments: {split}")
        tables.append(table)
        good = table[table["status"].eq("ok")].copy().reset_index(drop=True)
        features, labels, offsets, captured = [], [], [0], 0
        for row in good.to_dict("records"):
            with np.load(row["embedding_path"], allow_pickle=False) as arrays:
                x, y = arrays["features"], arrays[target_key]
                if (x.ndim != 2 or x.shape[1] != 768 or not (1 if audio else 4) <= len(x) <= 16
                        or y.shape != (len(x),) or not np.isfinite(x).all()
                        or not np.isin(y, [0, 1]).all() or int(arrays[clip_key]) != row["label"]
                        or (row["label"] == 0 and y.any())):
                    raise ValueError(f"Invalid {unit} features/labels: {row['source_file']}")
                features.append(x.astype(np.float32))
                labels.append(y.astype(np.int64))
                offsets.append(offsets[-1] + len(y))
                captured += int(row["label"] == 1 and y.any())
        if not features or set(good["label"]) != {0, 1}:
            raise ValueError(f"Both {modality} clip classes required in {split}")
        x, y = np.concatenate(features), np.concatenate(labels)
        if set(y) != {0, 1}:
            raise ValueError(f"Both unmodified and edited {unit} labels required in {split}")
        datasets[split] = dict(features=torch.from_numpy(x), labels=torch.from_numpy(y),
                               offsets=np.array(offsets), clips=good, modality=modality)
        coverage[split] = dict(total_clips=len(table), usable_clips=len(good),
                               **{f"{unit}_counts": np.bincount(y, minlength=2).tolist(),
                                  f"{modality}_fake_clips": int(good["label"].sum()),
                                  f"captured_{modality}_fake_clips": captured})
    combined = pd.concat(tables, ignore_index=True)
    if (combined["source_file"].duplicated().any() or combined["file_path"].duplicated().any()
            or combined.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("Repeated videos or source families cross train/validation")
    return datasets, coverage


@torch.inference_mode()
def sample_scores(head, dataset, device):
    head.eval()
    scores = np.concatenate([head(batch.to(device)).softmax(-1)[:, 1].cpu().numpy()
                             for batch in dataset["features"].split(512)])
    if not np.isfinite(scores).all():
        raise ValueError("Head returned non-finite scores")
    return scores


def evaluate(head, dataset, device):
    scores = sample_scores(head, dataset, device)
    clip_scores = np.maximum.reduceat(scores, dataset["offsets"][:-1])
    unit = "window" if dataset.get("modality") == "audio" else "frame"
    return {f"{unit}_auc": float(roc_auc_score(dataset["labels"].numpy(), scores)),
            "clip_auc": float(roc_auc_score(dataset["clips"]["label"], clip_scores))}, clip_scores


def training_epochs(head, features, labels, device, epochs):
    """Shared head-only updates; callers handle validation or fixed OOF epochs."""
    if epochs == 0:
        return
    counts = torch.bincount(labels, minlength=2).float()
    if (counts == 0).any():
        raise ValueError("Head training requires both sample classes")
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts)).to(device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
    loader = DataLoader(TensorDataset(features, labels), batch_size=min(128, len(labels)),
                        shuffle=True, drop_last=True)
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum, samples = 0., 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            samples += len(y)
        yield epoch, loss_sum / samples


def train(cache_dir, checkpoint_path, output_dir, epochs=15, modality="visual"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    provenance = json.loads((cache_dir / "provenance.json").read_text())
    audio = modality == "audio"
    unit = "window" if audio else "frame"
    feature_format = "frozen Wav2Vec2 weighted attentive pooling" if audio else "raw projected CLIP, not L2-normalized"
    if (provenance["checkpoint_sha256"] != sha256(checkpoint_path)
            or provenance["feature_format"] != feature_format):
        raise ValueError("The embedding cache must match the original head and feature format")
    data, coverage = load_cache(cache_dir, modality)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = make_head(checkpoint).to(device)
    baseline, baseline_scores = evaluate(head, data["val"], device)
    best, best_scores, best_state, best_epoch = baseline, baseline_scores.copy(), copy.deepcopy(head.state_dict()), 0
    print(f"Training only the {modality} classifier on {device}; cached embeddings remain fixed.", flush=True)
    print("Original head validation:", baseline, flush=True)
    stale, history = 0, []
    for epoch, loss in training_epochs(head, data["train"]["features"], data["train"]["labels"], device, epochs):
        validation, scores = evaluate(head, data["val"], device)
        history.append(dict(epoch=epoch, loss=loss, **validation))
        print(f"Epoch {epoch:02d}: loss={loss:.4f}, val {unit} AUC={validation[f'{unit}_auc']:.4f}, "
              f"val {modality}-clip AUC={validation['clip_auc']:.4f}", flush=True)
        if validation["clip_auc"] > best["clip_auc"] + 1e-4:
            best, best_scores, best_state, best_epoch = validation, scores.copy(), copy.deepcopy(head.state_dict()), epoch
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            print("Early stopping: validation clip AUC stopped improving.", flush=True)
            break
    labels = data["val"]["clips"]["label"].to_numpy()
    cutoff = select_balanced_threshold(labels, best_scores)
    validation_metrics = compute_binary_metrics(labels, best_scores, cutoff)
    validation_metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, best_scores >= cutoff))
    report = dict(best_epoch=best_epoch, improved=best_epoch > 0, baseline_validation=baseline,
                  selected_validation=best, validation_clip_metrics=validation_metrics,
                  video_decision_threshold=cutoff, threshold_source="validation", coverage=coverage, history=history,
                  scope=("Validation only; max over 4-second audio windows with 2-second hop; window scores are correlated."
                         if audio else "Validation only; max over 16 sampled face frames. The clip cutoff is not an individual-photo cutoff. Frame samples are correlated.")
                        + " No new held-out result or general AI-generator reliability claim is established.")
    keys = (("model_name", "hidden_dim", "dropout", "sample_rate", "duration", "class_names") if audio else
            ("model_name", "head_type", "hidden_dim", "dropout", "image_size", "class_names"))
    export = {key: checkpoint[key] for key in keys}
    export.update(head_state_dict={key: value.cpu().clone() for key, value in best_state.items()},
                  adaptation_dataset="LAV-DF" if best_epoch > 0 else None, validation_dataset="LAV-DF",
                  best_epoch=best_epoch, video_decision_threshold=cutoff,
                  threshold_source="validation", threshold_scope="max_over_16_face_samples",
                  provenance=provenance, coverage=coverage)
    if audio:
        report["clip_decision_threshold"] = report.pop("video_decision_threshold")
        export.pop("video_decision_threshold")
        head.load_state_dict(best_state)
        windows = sample_scores(head, data["val"], device)
        window_cutoff = select_balanced_threshold(data["val"]["labels"].numpy(), windows)
        report["window_decision_threshold"] = window_cutoff
        report["validation_window_metrics"] = compute_binary_metrics(data["val"]["labels"].numpy(), windows, window_cutoff)
        export["head_state_dict"] = {key: value.cpu().clone() for key, value in checkpoint["head_state_dict"].items()
                                     if not key.startswith("head.")}
        export["head_state_dict"].update({f"head.{key}": value.cpu().clone() for key, value in best_state.items()})
        export.update(decision_threshold=window_cutoff, clip_decision_threshold=cutoff, threshold_scope="4_second_window")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / ("audio_lavdf_head.pt" if audio else "image_lavdf_head.pt")
    if destination.resolve() == checkpoint_path.resolve():
        raise ValueError("Use a separate destination to preserve the original checkpoint")
    temporary = destination.with_suffix(".tmp")
    torch.save(export, temporary)
    temporary.replace(destination)
    frame = data["val"]["clips"].copy()
    frame["baseline_score"], frame["adapted_score"] = baseline_scores, best_scores
    frame.to_csv(output_dir / f"{modality}_val_predictions.csv", index=False)
    (output_dir / f"{modality}_head_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2), flush=True)
    print(f"Saved selected head and validation results to {output_dir}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--modality", choices=["visual", "audio"], default="visual")
    args = parser.parse_args()
    train(args.cache_dir, args.checkpoint, args.output_dir, args.epochs, args.modality)
