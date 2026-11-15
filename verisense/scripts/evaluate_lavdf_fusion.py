"""Evaluate the frozen LAV-DF fusion candidate; never fit or select on test."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fusion.build_features import MultimodalScorer, read_manifests, save_json, sha256
from src.fusion.model import FEATURE_COLUMNS, LateFusionClassifier
from src.fusion.train_fusion import load_features, metrics
from src.video.face_extractor import FaceExtractor


def evaluate(base, checkpoint_dir, feature_dir, output_dir):
    run = base / "checkpoints/fusion_run2"
    fusion_path = run / "fusion_best.pt"
    checkpoint = torch.load(fusion_path, map_location="cpu", weights_only=True)
    fusion = LateFusionClassifier(checkpoint).eval()
    development = json.loads((run / "fusion_metrics.json").read_text())
    lineage = checkpoint["provenance"]
    if (lineage.get("training_predictions") != "grouped_out_of_fold"
            or "test_metrics" in development
            or development["decision_threshold"] != fusion.decision_threshold
            or development["provenance"] != lineage):
        raise ValueError("Expected the frozen validation-only OOF fusion candidate")
    manifest_dir = base / "fusion/lavdf_subset/manifests"
    for split in ("train", "val"):
        if sha256(manifest_dir / f"{split}.csv") != lineage["manifests"][split]:
            raise ValueError("Development source assignments changed after fusion training")
    # Check family separation using every manifest, including rejected examples.
    plan = read_manifests(manifest_dir)
    plan = plan[plan["split"].eq("test")].copy()
    paths = {"image_head.pt": base / "checkpoints/visual_lavdf_run1/image_lavdf_head.pt",
             "audio_head.pt": base / "checkpoints/audio_lavdf_run1/audio_lavdf_head.pt",
             "video_best.pt": checkpoint_dir / "video_best.pt"}
    if any(sha256(path) != lineage["checkpoints"][name] for name, path in paths.items()):
        raise ValueError("A branch differs from the weights frozen during fusion training")
    extractor = FaceExtractor()
    backend = "mediapipe" if extractor.use_mediapipe else "haar"
    root = Path(__file__).resolve().parents[1]
    visual, audio = lineage["caches"]["visual"], lineage["caches"]["audio"]
    if (visual["face_backend"] != backend
            or visual["face_code_sha256"] != sha256(root / "src/video/face_extractor.py")
            or audio["audio_code_sha256"] != sha256(root / "src/audio/model.py")
            or audio["preprocessing_sha256"] != sha256(root / "src/audio/preprocessing.py")):
        raise ValueError("Face detector or speech preprocessing differs from training caches")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode test soundtracks")
    provenance = dict(feature_columns=list(FEATURE_COLUMNS), checkpoints=lineage["checkpoints"],
        manifests={"test": sha256(manifest_dir / "test.csv")}, fusion_sha256=sha256(fusion_path),
        development_metrics_sha256=sha256(run / "fusion_metrics.json"), face_backend=backend,
        code={str(path.relative_to(root)): sha256(path) for path in [
            Path(__file__), root / "src/fusion/build_features.py", root / "src/image/model.py",
            root / "src/video/model.py", root / "src/video/face_extractor.py",
            root / "src/audio/model.py", root / "src/audio/preprocessing.py"]})
    metadata = feature_dir / "run_metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != provenance:
        raise ValueError("Frozen inputs changed; use a new test feature directory")
    save_json(metadata, provenance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Frozen test evaluation on {device}; cutoff={fusion.decision_threshold}; no training.", flush=True)
    scorer, results = None, []
    # The scorer accepts its usual three filenames; temporary copies preserve originals.
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for name, path in paths.items():
            shutil.copy2(path, folder / name)
        for row in tqdm(plan.to_dict("records"), desc="Scoring frozen test clips"):
            cache = feature_dir / "clips" / f"{hashlib.sha256(row['source_file'].encode()).hexdigest()}.json"
            digest = sha256(row["file_path"])
            result = json.loads(cache.read_text()) if cache.exists() else {}
            if result.get("file_sha256") != digest:
                if scorer is None:
                    scorer = MultimodalScorer.from_checkpoints(folder, device, extractor, ffmpeg)
                try:
                    result = dict(scorer.score(Path(row["file_path"])), status="ok", file_sha256=digest)
                except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                    result = dict(status="error", error=str(exc), file_sha256=digest)
                    tqdm.write(f"Rejected {row['source_file']}: {exc}")
                save_json(cache, result)
            results.append(dict(row, **result))
    table = pd.DataFrame(results)
    table.to_csv(feature_dir / "test_features.csv", index=False)
    tables, coverage, _ = load_features(feature_dir, manifest_dir, splits=("test",))
    good = tables["test"].copy()
    with torch.inference_mode():
        scores = fusion(good[list(FEATURE_COLUMNS)].to_numpy()).numpy()
    good["fusion_score"], good["fusion_prediction"] = scores, (scores >= fusion.decision_threshold).astype(int)
    report = dict(decision_threshold=fusion.decision_threshold, selected_on="validation",
        test_metrics=metrics(good, scores, fusion.decision_threshold), coverage=coverage,
        test_majority_class_accuracy=float(good["label"].value_counts(normalize=True).max()),
        baselines={name: dict(threshold=development["baselines"][name]["threshold"],
            test_metrics=metrics(good, good[name].to_numpy(), development["baselines"][name]["threshold"]))
            for name in FEATURE_COLUMNS}, provenance=provenance,
        scope="Frozen candidate on the existing LAV-DF test subset, previously used for the first fusion baseline. "
              "Usable clips only; rejections are abstentions. Source families are separated; speaker separation is unverified. "
              "Scores are uncalibrated; this does not establish reliability on modern AI generators.")
    output_dir.mkdir(parents=True, exist_ok=True)
    good.to_csv(output_dir / "fusion_test_predictions.csv", index=False)
    save_json(output_dir / "fusion_test_metrics.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("provenance", "baselines")}, indent=2))
    comparison = {name: {key: item["test_metrics"][key] for key in ("accuracy", "balanced_accuracy", "auc", "f1")}
                  for name, item in report["baselines"].items()}
    comparison["fusion"] = {key: report["test_metrics"][key] for key in ("accuracy", "balanced_accuracy", "auc", "f1")}
    print("Test comparison; weights and all cutoffs frozen from development:")
    print(pd.DataFrame(comparison).T.to_string())
    print(f"Saved test report to {output_dir}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.base_dir, args.checkpoint_dir, args.feature_dir, args.output_dir)
