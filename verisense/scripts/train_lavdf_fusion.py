"""Grouped out-of-fold head predictions, followed by validation-only late fusion."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lavdf_head import load_cache, make_head, sample_scores, sha256, training_epochs
from src.common.utils import set_seed
from src.fusion.model import FEATURE_COLUMNS
from src.fusion.train_fusion import train_late_fusion


def save_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def subset(data, clip_mask):
    indices = np.flatnonzero(clip_mask)
    if not len(indices):
        raise ValueError("Fold has no usable clips for a modality")
    spans = [(data["offsets"][i], data["offsets"][i + 1]) for i in indices]
    return dict(features=torch.cat([data["features"][start:end] for start, end in spans]),
                labels=torch.cat([data["labels"][start:end] for start, end in spans]),
                offsets=np.r_[0, np.cumsum([end - start for start, end in spans])],
                clips=data["clips"].iloc[indices].reset_index(drop=True), modality=data["modality"])


def predict(head, data, device):
    scores = sample_scores(head, data, device)
    starts = data["offsets"][:-1]
    frame = data["clips"][["source_file", "group_id"]].copy()
    frame["mean"] = np.add.reduceat(scores, starts) / np.diff(data["offsets"])
    frame["max"] = np.maximum.reduceat(scores, starts)
    return frame


def fit_fold(source, training, held, epochs, device, seed):
    if set(training["clips"]["group_id"]).intersection(held["clips"]["group_id"]):
        raise ValueError("A source family crosses the OOF training/held-out boundary")
    set_seed(seed)
    head = make_head(source).to(device)
    # Fixed selected epoch count: no held-out fold labels participate in selection.
    for _ in training_epochs(head, training["features"], training["labels"], device, epochs):
        pass
    return predict(head, held, device)


def validate_scores(frame, expected):
    if (not {"source_file", "group_id", "mean", "max"}.issubset(frame.columns)
            or frame["source_file"].duplicated().any() or set(frame["source_file"]) != set(expected["source_file"])):
        raise ValueError("Incomplete or repeated OOF prediction records")
    left = frame.set_index("source_file").loc[expected["source_file"]]
    if left["group_id"].tolist() != expected["group_id"].tolist():
        raise ValueError("OOF source assignments changed")
    values = frame[["mean", "max"]].to_numpy()
    if (not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all()
            or not (frame["mean"] <= frame["max"] + 1e-7).all()):
        raise ValueError("Invalid head prediction scores")


def rebuild(base, checkpoint_dir, feature_dir, output_dir, folds=3):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_dir, manifest_dir = base / "fusion/lavdf_features_v1", base / "fusion/lavdf_subset/manifests"
    old_meta = json.loads((old_dir / "run_metadata.json").read_text())
    datasets, sources, adapted, cache_meta = {}, {}, {}, {}
    final_paths = {"visual": base / "checkpoints/visual_lavdf_run1/image_lavdf_head.pt",
                   "audio": base / "checkpoints/audio_lavdf_run1/audio_lavdf_head.pt"}
    source_names = {"visual": "image_head.pt", "audio": "audio_head.pt"}
    for modality in ("visual", "audio"):
        cache = base / f"adaptation/lavdf_{modality}_v1"
        datasets[modality], _, cache_meta[modality] = load_modality(cache, modality)
        source_path = checkpoint_dir / source_names[modality]
        sources[modality] = torch.load(source_path, map_location="cpu", weights_only=True)
        adapted[modality] = torch.load(final_paths[modality], map_location="cpu", weights_only=True)
        if (sha256(source_path) != cache_meta[modality]["checkpoint_sha256"]
                or adapted[modality]["provenance"] != cache_meta[modality]
                or old_meta["checkpoints"][source_names[modality]] != sha256(source_path)):
            raise ValueError(f"Source, adapted head or embedding lineage differs: {modality}")
        epoch = adapted[modality]["best_epoch"]
        if type(epoch) is not int or not 0 <= epoch <= 15:
            raise ValueError("Expected a selected adaptation epoch between zero and 15")
        architecture = ("model_name", "hidden_dim", "dropout", "class_names") + (
            ("head_type", "image_size") if modality == "visual" else ("sample_rate", "duration"))
        if any(sources[modality][key] != adapted[modality][key] for key in architecture):
            raise ValueError("The OOF learner must match the selected full-head architecture")
        if modality == "audio" and any(not torch.equal(value, adapted[modality]["head_state_dict"][key])
                for key, value in sources[modality]["head_state_dict"].items() if not key.startswith("head.")):
            raise ValueError("Audio layer/pooling weights changed; regenerate speech embeddings")
    if old_meta["checkpoints"]["video_best.pt"] != sha256(checkpoint_dir / "video_best.pt"):
        raise ValueError("The preserved temporal scores require their original checkpoint")
    groups = datasets["audio"]["train"]["clips"]["group_id"]
    if not 2 <= folds <= groups.nunique():
        raise ValueError("Need at least two folds and enough source families")
    assignments = datasets["audio"]["train"]["clips"][["source_file", "group_id"]].copy()
    assignments["fold"] = -1
    for fold, (_, held) in enumerate(GroupKFold(folds).split(assignments, groups=groups)):
        assignments.loc[held, "fold"] = fold
    group_fold = assignments.groupby("group_id")["fold"].first().to_dict()
    provenance = dict(feature_columns=list(FEATURE_COLUMNS), training_predictions="grouped_out_of_fold",
        folds=folds, selected_epochs={key: value["best_epoch"] for key, value in adapted.items()},
        checkpoints={"image_head.pt": sha256(final_paths["visual"]), "audio_head.pt": sha256(final_paths["audio"]),
                     "video_best.pt": sha256(checkpoint_dir / "video_best.pt")},
        source_checkpoints={name: sha256(checkpoint_dir / name) for name in source_names.values()},
        manifests={split: sha256(manifest_dir / f"{split}.csv") for split in ("train", "val")},
        caches=cache_meta, feature_tables={split: sha256(old_dir / f"{split}_features.csv") for split in ("train", "val")},
        cache_manifests={modality: {split: sha256(base / f"adaptation/lavdf_{modality}_v1/{split}.csv")
                                   for split in ("train", "val")} for modality in ("visual", "audio")},
        code_sha256=sha256(__file__), head_training_code_sha256=sha256(Path(__file__).with_name("train_lavdf_head.py")))
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = feature_dir / "run_metadata.json"
    if metadata_file.exists() and json.loads(metadata_file.read_text()) != provenance:
        raise ValueError("OOF inputs or code changed; use a new feature directory")
    temporary = metadata_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(provenance, indent=2))
    temporary.replace(metadata_file)
    save_csv(feature_dir / "oof_assignments.csv", assignments)
    print(f"Head-only OOF fitting on {device}: {folds} folds, {groups.nunique()} source families.", flush=True)
    pooled = {"train": {}, "val": {}}
    for modality in ("visual", "audio"):
        data = datasets[modality]["train"]
        clip_folds = data["clips"]["group_id"].map(group_fold)
        if clip_folds.isna().any():
            raise ValueError("Visual/audio source families do not match")
        pieces = []
        for fold in range(folds):
            held = subset(data, clip_folds.eq(fold).to_numpy())
            path = feature_dir / "folds" / f"{modality}_{fold}.csv"
            if path.exists():
                frame = pd.read_csv(path, dtype={"source_file": str, "group_id": str})
            else:
                training = subset(data, clip_folds.ne(fold).to_numpy())
                print(f"{modality}: fold {fold + 1}/{folds}, train families={training['clips'].group_id.nunique()}, "
                      f"held-out families={held['clips'].group_id.nunique()}", flush=True)
                frame = fit_fold(sources[modality], training, held, adapted[modality]["best_epoch"], device, 42 + fold)
                save_csv(path, frame)
            validate_scores(frame, held["clips"])
            pieces.append(frame)
        pooled["train"][modality] = pd.concat(pieces, ignore_index=True)
        head = make_head(adapted[modality]).to(device)
        pooled["val"][modality] = predict(head, datasets[modality]["val"], device)
    for split in ("train", "val"):
        frame = pd.read_csv(old_dir / f"{split}_features.csv", dtype={"group_id": str, "source_file": str})
        if "error" not in frame:
            frame["error"] = pd.NA
        frame = frame.drop(columns=["image_mean", "image_max", "audio_mean", "audio_max"], errors="ignore")
        for modality, prefix in (("visual", "image"), ("audio", "audio")):
            scores = pooled[split][modality].rename(columns={"mean": f"{prefix}_mean", "max": f"{prefix}_max"})
            frame = frame.merge(scores, on=["source_file", "group_id"], how="left", validate="one_to_one")
        available = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
        failed = frame["status"].ne("ok") | ~available
        frame.loc[failed, "status"] = "error"
        frame.loc[failed & frame["error"].isna(), "error"] = "A required cached modality is unavailable"
        save_csv(feature_dir / f"{split}_features.csv", frame)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/fusion.yaml").read_text())
    return train_late_fusion(feature_dir, manifest_dir, output_dir, config, validation_only=True)


def load_modality(cache, modality):
    datasets, coverage = load_cache(cache, modality)
    return datasets, coverage, json.loads((cache / "provenance.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.base_dir, args.checkpoint_dir, args.feature_dir, args.output_dir)
