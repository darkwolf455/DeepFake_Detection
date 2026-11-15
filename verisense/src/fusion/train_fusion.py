"""Train small real-data late fusion; preserve official source partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.common.metrics import compute_binary_metrics, select_balanced_threshold as select_threshold
from src.fusion.model import FEATURE_COLUMNS, LateFusionClassifier


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(frame, scores, threshold):
    result = compute_binary_metrics(frame["label"].to_numpy(), scores, threshold)
    predictions = np.asarray(scores) >= threshold
    result["balanced_accuracy"] = float(balanced_accuracy_score(frame["label"], predictions))
    result["category_accuracy"] = {
        category: {"n": int(mask.sum()), "accuracy": float((predictions[mask] == frame.loc[mask, "label"]).mean())}
        for category in sorted(frame["category"].unique())
        for mask in [frame["category"].eq(category).to_numpy()]
    }
    return result


def load_features(features_dir, manifest_dir, splits=("train", "val", "test")):
    provenance = json.loads((features_dir / "run_metadata.json").read_text())
    if (provenance["feature_columns"] != list(FEATURE_COLUMNS)
            or set(provenance["checkpoints"]) != {"image_head.pt", "video_best.pt", "audio_head.pt"}):
        raise ValueError("Expected paired predictions from the three trained branches")
    usable, coverage, raw_tables = {}, {}, []
    categories = {(False, False): "real", (True, False): "video_only",
                  (False, True): "audio_only", (True, True): "both"}
    for split in splits:
        manifest = manifest_dir / f"{split}.csv"
        if sha256(manifest) != provenance["manifests"][split]:
            raise ValueError(f"The {split} manifest changed after feature extraction")
        types = {"source_file": str, "group_id": str}
        plan = pd.read_csv(manifest, dtype=types).set_index("source_file")
        raw = pd.read_csv(features_dir / f"{split}_features.csv", dtype=types).set_index("source_file")
        if not plan.index.is_unique or not raw.index.is_unique or set(raw.index) != set(plan.index):
            raise ValueError(f"Incomplete or duplicate {split} features; finish extraction first")
        raw = raw.loc[plan.index]
        columns = ["file_path", "group_id", "label", "split", "category", "modify_video", "modify_audio"]
        if not raw[columns].equals(plan[columns]) or not raw["split"].eq(split).all():
            raise ValueError(f"Feature labels/source assignments disagree with {split} manifest")
        if raw[columns].isna().any().any() or not raw["status"].isin(["ok", "error"]).all():
            raise ValueError(f"Invalid {split} labels or status")
        for (video, audio), category in categories.items():
            mask = raw["category"].eq(category)
            if not (raw.loc[mask, "modify_video"].eq(video) & raw.loc[mask, "modify_audio"].eq(audio)).all():
                raise ValueError("Categories disagree with annotation modification flags")
        labels = raw["modify_video"].eq(True) | raw["modify_audio"].eq(True)
        if not raw["label"].eq(labels.astype(int)).all():
            raise ValueError("Labels disagree with annotation modification flags")
        good = raw[raw["status"].eq("ok")].copy().reset_index()
        values = good[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if (set(good["label"]) != {0, 1} or set(good["category"]) != set(categories.values())
                or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all()
                or not good["face_frames"].ge(4).all() or not good["scored_audio_windows"].ge(1).all()):
            raise ValueError(f"Unusable {split} features; inspect extraction failures")
        usable[split] = good
        coverage[split] = {"total": len(raw), "scored": len(good), "rejected": len(raw) - len(good),
                           "coverage": len(good) / len(raw), "categories": {}}
        for category, group in raw.groupby("category"):
            coverage[split]["categories"][category] = {
                "total": len(group), "scored": int(group["status"].eq("ok").sum())}
        raw_tables.append(raw.reset_index())
    all_rows = pd.concat(raw_tables, ignore_index=True)
    if (all_rows["source_file"].duplicated().any() or all_rows["file_path"].duplicated().any()
            or all_rows.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("Duplicate videos or related sources cross partitions")
    return usable, coverage, provenance


def train_late_fusion(features_dir, manifest_dir, output_dir, config, validation_only=False):
    splits = ("train", "val") if validation_only else ("train", "val", "test")
    tables, coverage, provenance = load_features(features_dir, manifest_dir, splits)
    train, val = tables["train"], tables["val"]
    scaler = StandardScaler().fit(train[list(FEATURE_COLUMNS)])
    x_train, x_val = (scaler.transform(frame[list(FEATURE_COLUMNS)]) for frame in (train, val))
    trials, best = [], None
    for strength in sorted(config["regularization_candidates"]):
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("Regularization candidates must be positive and finite")
        classifier = LogisticRegression(C=strength, class_weight="balanced", solver="lbfgs",
                                        max_iter=config["max_iter"], random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(x_train, train["label"])
        score = float(roc_auc_score(val["label"], classifier.predict_proba(x_val)[:, 1]))
        trials.append({"C": float(strength), "validation_auc": score})
        if best is None or score > best[0]:
            best = score, classifier
    if best is None:
        raise ValueError("At least one regularization candidate is required")
    classifier = best[1]
    threshold = select_threshold(val["label"], classifier.predict_proba(x_val)[:, 1])
    checkpoint = {
        "model_type": "standardized_logistic_regression", "feature_columns": list(FEATURE_COLUMNS),
        "class_names": ["real", "manipulated"], "decision_threshold": threshold,
        "threshold_source": "validation", "calibrated": False,
        "mean": torch.from_numpy(scaler.mean_.copy()), "scale": torch.from_numpy(scaler.scale_.copy()),
        "weight": torch.from_numpy(classifier.coef_[0].copy()),
        "bias": torch.from_numpy(classifier.intercept_.copy()), "C": float(classifier.C),
        "provenance": provenance, "coverage": coverage,
        "feature_tables": {split: sha256(features_dir / f"{split}_features.csv") for split in tables},
    }
    model = LateFusionClassifier(checkpoint).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "fusion_best.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output_dir / "fusion_best.pt")  # Freeze choices before test evaluation.
    report = {"selected_on": "validation", "decision_threshold": threshold, "C": float(classifier.C),
              "candidates": trials, "coverage": coverage, "baselines": {},
              "provenance": provenance, "feature_tables": checkpoint["feature_tables"],
              "scope": "Metrics cover usable clips only. Rejections are abstentions, not correct predictions. "
                       "Source-video splits are preserved; speaker separation is unverified. Scores are uncalibrated."}
    evaluation_split = "val" if validation_only else "test"
    evaluation = tables[evaluation_split]
    for split in (("val",) if validation_only else ("val", "test")):
        frame = tables[split].copy()
        with torch.inference_mode():
            scores = model(frame[list(FEATURE_COLUMNS)].to_numpy()).numpy()
        report[f"{split}_metrics"] = metrics(frame, scores, threshold)
        frame["fusion_score"], frame["fusion_prediction"] = scores, (scores >= threshold).astype(int)
        frame.to_csv(output_dir / f"fusion_{split}_predictions.csv", index=False)
    for column in FEATURE_COLUMNS:
        cutoff = select_threshold(val["label"], val[column])
        report["baselines"][column] = {"threshold": cutoff,
            f"{evaluation_split}_metrics": metrics(evaluation, evaluation[column].to_numpy(), cutoff)}
    report[f"{evaluation_split}_majority_class_accuracy"] = float(evaluation["label"].value_counts(normalize=True).max())
    if validation_only:
        report["scope"] += " Validation development result only; no test split was read."
    (output_dir / "fusion_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("Coverage (rejected clips receive no fusion decision):")
    print(pd.DataFrame(coverage).T.drop(columns="categories").to_string())
    print(json.dumps({key: report[key] for key in
                     ("C", "decision_threshold", "val_metrics", "test_metrics") if key in report}, indent=2, allow_nan=False))
    comparison = {name: dict(threshold=result["threshold"], **{
        key: result[f"{evaluation_split}_metrics"][key] for key in ("accuracy", "balanced_accuracy", "auc", "f1")
    }) for name, result in report["baselines"].items()}
    comparison["fusion"] = dict(threshold=threshold, **{
        key: report[f"{evaluation_split}_metrics"][key] for key in ("accuracy", "balanced_accuracy", "auc", "f1")})
    print(f"{evaluation_split.title()} comparison; all cutoffs selected on validation:")
    print(pd.DataFrame(comparison).T.to_string())
    print(report["scope"])
    print(f"Saved fusion_best.pt, metrics and predictions to {output_dir}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/fusion.yaml")
    parser.add_argument("--validation-only", action="store_true", help="Develop without opening the test split")
    args = parser.parse_args()
    train_late_fusion(args.features_dir, args.manifest_dir, args.output_dir,
                      yaml.safe_load(args.config.read_text()), args.validation_only)


if __name__ == "__main__":
    main()
