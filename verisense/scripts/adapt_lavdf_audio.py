"""Cache frozen speech embeddings, then adapt the audio head on LAV-DF train/val."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.audio.model import Wav2Vec2DeepfakeDetector
from src.audio.preprocessing import audio_windows, decode_audio
from train_lavdf_head import sha256, train


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def window_labels(metadata, bounds):
    periods, duration = metadata["fake_periods"], metadata["duration"]
    if not np.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError("Invalid annotated audio duration")
    for period in periods:
        if (len(period) != 2 or not np.isfinite(period).all()
                or not 0 <= period[0] < period[1] <= duration + .1):
            raise ValueError(f"Invalid edit periods: {metadata['file']}")
    if metadata["modify_audio"] and not periods:
        raise ValueError("Audio manipulation has no annotated interval")
    return np.array([int(metadata["modify_audio"] and any(start < b and end > a for a, b in periods))
                     for start, end in bounds], dtype=np.int64)


@torch.inference_mode()
def extract(path, metadata, model, device, ffmpeg):
    wave = decode_audio(path, ffmpeg)
    features, bounds = [], []
    for start, window in audio_windows(wave):
        if np.max(np.abs(window)) <= 1e-6:
            continue
        tensor = torch.from_numpy(window).unsqueeze(0).to(device)
        features.append(model.extract_features(tensor)[0].cpu().numpy())
        bounds.append((start, min(start + 4., len(wave) / 16000)))
    if not features:
        raise ValueError("No non-silent audio window")
    labels = window_labels(metadata, bounds)
    if metadata["modify_audio"] and not labels.any():
        raise ValueError("No annotated audio edit inside decoded, non-silent windows")
    features = np.stack(features).astype(np.float32)
    if features.shape != (len(bounds), 768) or not np.isfinite(features).all():
        raise ValueError("Invalid speech embeddings")
    return dict(features=features, bounds=np.array(bounds, dtype=np.float32), window_labels=labels,
                audio_label=np.array(int(metadata["modify_audio"]), dtype=np.int64))


def prepare(subset_dir, checkpoint, cache_dir):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode the existing MP4 soundtracks")
    annotation_file = subset_dir / "selected_metadata.json"
    annotations = {row["file"]: row for row in json.loads(annotation_file.read_text())}
    plans = []
    for split in ("train", "val"):
        plan = pd.read_csv(subset_dir / "manifests" / f"{split}.csv", dtype={"group_id": str})
        if plan.empty or not plan["split"].eq(split).all():
            raise ValueError(f"Invalid {split} manifest")
        for row in plan.to_dict("records"):
            metadata = annotations[row["source_file"]]
            if (metadata["split"] != ("dev" if split == "val" else split)
                    or (metadata["original"] or metadata["file"]) != row["group_id"]
                    or metadata["modify_audio"] != row["modify_audio"]
                    or metadata["modify_video"] != row["modify_video"]
                    or row["label"] != int(metadata["modify_audio"] or metadata["modify_video"])):
                raise ValueError("Annotations disagree with saved source assignments")
            window_labels(metadata, [])
            if not Path(row["file_path"]).is_file():
                raise FileNotFoundError(row["file_path"])
        plans.append(plan)
    plan = pd.concat(plans, ignore_index=True)
    if (plan["source_file"].duplicated().any() or plan["file_path"].duplicated().any()
            or plan.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("Duplicate clips or source families cross train/validation")
    head = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = dict(checkpoint_sha256=sha256(checkpoint), annotation_sha256=sha256(annotation_file),
                      feature_format="frozen Wav2Vec2 weighted attentive pooling",
                      protocol="4s windows, 2s hop, end covered, 16kHz mono, RMS .1",
                      code_sha256=sha256(__file__),
                      audio_code_sha256=sha256(Path(__file__).resolve().parents[1] / "src/audio/model.py"),
                      preprocessing_sha256=sha256(Path(__file__).resolve().parents[1] / "src/audio/preprocessing.py"),
                      manifests={split: sha256(subset_dir / "manifests" / f"{split}.csv") for split in ("train", "val")})
    path = cache_dir / "provenance.json"
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError("Audio inputs or extraction code changed; use a new cache directory")
    save_json(path, provenance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Caching train/val speech features on {device}; no face requirement.", flush=True)
    model, results = None, []
    for row in tqdm(plan.to_dict("records"), desc="Caching speech embeddings"):
        key = hashlib.sha256(row["source_file"].encode()).hexdigest()
        destination = cache_dir / "clips" / f"{key}.npz"
        record = destination.with_suffix(".json")
        digest = sha256(row["file_path"])
        result = json.loads(record.read_text()) if record.exists() else {}
        if result.get("file_sha256") != digest or (result.get("status") == "ok" and not destination.exists()):
            if model is None:
                model = Wav2Vec2DeepfakeDetector.from_checkpoint(head).to(device).eval().requires_grad_(False)
            try:
                arrays = extract(Path(row["file_path"]), annotations[row["source_file"]], model, device, ffmpeg)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(destination)
                result = dict(status="ok", embedding_path=str(destination), windows=len(arrays["window_labels"]),
                              edited_windows=int(arrays["window_labels"].sum()), file_sha256=digest)
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                result = dict(status="error", error=str(exc), file_sha256=digest)
                tqdm.write(f"Rejected {row['source_file']}: {exc}")
            save_json(record, result)
        results.append(dict(row, **result, overall_label=row["label"], label=int(row["modify_audio"])))
    table = pd.DataFrame(results)
    for split, frame in table.groupby("split"):
        frame.to_csv(cache_dir / f"{split}.csv", index=False)
    print(table.groupby(["split", "category", "status"]).size().to_string(), flush=True)
    del model  # Release the large backbone before fitting the small head.
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.subset_dir, args.checkpoint, args.cache_dir)
    train(args.cache_dir, args.checkpoint, args.output_dir, modality="audio")
