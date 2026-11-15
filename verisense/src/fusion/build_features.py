"""Cache real paired audio/visual predictions for learned late fusion.

No labels or manipulation timestamps choose frames/audio windows.
Requires three trained checkpoints; failures stay explicit, never imputed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from src.audio.model import Wav2Vec2DeepfakeDetector
from src.audio.preprocessing import audio_windows, decode_audio
from src.image.model import CLIPDeepfakeDetector
from src.video.face_extractor import FaceExtractor, sample_video_frames
from src.video.model import VideoTemporalDetector

FEATURE_COLUMNS = ("image_mean", "image_max", "video_score", "audio_mean", "audio_max")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class MultimodalScorer:
    """Frozen branches; image and temporal heads share one CLIP pass."""

    def __init__(self, image, video, audio, device, face_extractor, ffmpeg):
        self.image, self.video, self.audio = image, video, audio
        self.device, self.face_extractor, self.ffmpeg = device, face_extractor, ffmpeg

    @classmethod
    def from_checkpoints(cls, directory, device, face_extractor, ffmpeg):
        image = torch.load(directory / "image_head.pt", map_location="cpu", weights_only=True)
        video = torch.load(directory / "video_best.pt", map_location="cpu", weights_only=True)
        audio = torch.load(directory / "audio_head.pt", map_location="cpu", weights_only=True)
        if image["class_names"] != ["real", "manipulated"] or image["image_size"] != 224:
            raise ValueError("Unsupported image labels or preprocessing")
        image_model = CLIPDeepfakeDetector(**{
            name: image[name] for name in ("model_name", "head_type", "hidden_dim", "dropout")
        })
        image_model.head.load_state_dict(image["head_state_dict"], strict=True)
        config = video["config"]["temporal_model"]
        if config["input_dim"] != 768 or config["max_frames"] != 16:
            raise ValueError("Expected the trained 768-dimensional, 16-frame temporal model")
        video_model = VideoTemporalDetector.from_config(config)
        video_model.load_state_dict(video["model_state_dict"], strict=True)
        audio_model = Wav2Vec2DeepfakeDetector.from_checkpoint(audio)
        models = [model.to(device).eval().requires_grad_(False)
                  for model in (image_model, video_model, audio_model)]
        return cls(*models, device, face_extractor, ffmpeg)

    @torch.inference_mode()
    def score(self, path):
        crops, sampled = [], 0
        for rgb, _ in sample_video_frames(path, max_frames=self.video.max_frames):
            sampled += 1
            faces = self.face_extractor.extract_faces(rgb)
            if len(faces) == 1:
                crops.append(faces[0]["cropped_face"])
        if len(crops) < 4:
            raise ValueError(f"Only {len(crops)} sampled frames contained exactly one face; need four")

        mean = torch.tensor([.48145466, .4578275, .40821073], device=self.device)[None, :, None, None]
        std = torch.tensor([.26862954, .26130258, .27577711], device=self.device)[None, :, None, None]
        projected = []
        for start in range(0, len(crops), 4):
            pixels = torch.from_numpy(np.stack(crops[start:start + 4])).to(self.device)
            pixels = pixels.permute(0, 3, 1, 2).float() / 255
            projected.append(self.image.backbone(pixel_values=(pixels - mean) / std).image_embeds)
        projected = torch.cat(projected)
        image_scores = self.image.head(projected).softmax(-1)[:, 1]
        features = F.normalize(projected, dim=-1).unsqueeze(0)
        mask = torch.ones(1, len(crops), dtype=torch.bool, device=self.device)
        logits, _ = self.video(features, mask=mask)

        waveform = decode_audio(path, self.ffmpeg)
        audio_scores = []
        windows = audio_windows(waveform)
        for _, window in windows:
            if np.max(np.abs(window)) <= 1e-6:
                continue
            tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
            audio_scores.append(float(self.audio(tensor).softmax(-1)[0, 1]))
        if not audio_scores:
            raise ValueError("No non-silent audio window")
        values = [float(image_scores.mean()), float(image_scores.max()),
                  float(logits.softmax(-1)[0, 1]), float(np.mean(audio_scores)), max(audio_scores)]
        if not np.isfinite(values).all() or not all(0 <= value <= 1 for value in values):
            raise ValueError("A trained branch returned invalid scores")
        return dict(zip(FEATURE_COLUMNS, values), sampled_frames=sampled, face_frames=len(crops),
                    audio_windows=len(windows), scored_audio_windows=len(audio_scores),
                    audio_seconds=len(waveform) / 16000)


def read_manifests(directory):
    import pandas as pd
    tables = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(directory / f"{split}.csv", dtype={"group_id": str, "source_file": str})
        required = {"file_path", "source_file", "group_id", "label", "split", "category",
                    "modify_video", "modify_audio", "duration"}
        if not required.issubset(frame) or frame[list(required)].isna().any().any():
            raise ValueError(f"Incomplete {split} manifest")
        if frame.empty or not frame["split"].eq(split).all() or set(frame["label"]) != {0, 1}:
            raise ValueError(f"Invalid split assignments or labels: {split}")
        expected = frame["modify_video"].eq(True) | frame["modify_audio"].eq(True)
        if not frame["label"].eq(expected.astype(int)).all():
            raise ValueError("Labels disagree with the annotation modification flags")
        if not frame["duration"].between(.1, 30).all():
            raise ValueError("This fusion experiment accepts clips up to 30 seconds")
        tables.append(frame)
    frame = pd.concat(tables, ignore_index=True)
    if frame["source_file"].duplicated().any() or frame["file_path"].duplicated().any():
        raise ValueError("Duplicate clips across manifests")
    if frame.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("Related source clips cross partitions")
    missing = [path for path in frame["file_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved clips: {missing[:3]}")
    return frame


def main():
    import pandas as pd
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Smoke check; zero processes all clips")
    args = parser.parse_args()
    if args.limit_per_split < 0:
        parser.error("--limit-per-split cannot be negative")
    frame = read_manifests(args.manifest_dir)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode video soundtracks; Colab provides it")
    extractor = FaceExtractor()
    source_root = Path(__file__).resolve().parents[1]
    metadata = {
        "feature_columns": list(FEATURE_COLUMNS), "visual_sampling": "uniform, at most 16 frames / first 30s",
        "audio_sampling": "4s windows, 2s hop, end covered; silent windows skipped",
        "face_backend": "mediapipe" if extractor.use_mediapipe else "haar",
        "checkpoints": {name: sha256(args.checkpoint_dir / name)
                        for name in ("image_head.pt", "video_best.pt", "audio_head.pt")},
        "manifests": {split: sha256(args.manifest_dir / f"{split}.csv") for split in ("train", "val", "test")},
        "code": {str(path.relative_to(source_root)): sha256(path) for path in [
            Path(__file__), source_root / "image/model.py", source_root / "video/model.py",
            source_root / "video/face_extractor.py", source_root / "audio/model.py",
            source_root / "audio/preprocessing.py",
        ]},
    }
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Models, manifests or extraction code changed. Use a new output directory.")
    save_json(metadata_path, metadata)
    if args.limit_per_split:
        frame = frame.groupby("split", sort=False).head(args.limit_per_split)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; clips: {len(frame)}; face detector: {metadata['face_backend']}", flush=True)
    scorer, results = None, []
    for row in tqdm(frame.to_dict("records"), desc="Caching paired predictions"):
        key = hashlib.sha256(row["source_file"].encode()).hexdigest()
        cache = args.output_dir / "clips" / f"{key}.json"
        digest = sha256(row["file_path"])
        result = json.loads(cache.read_text()) if cache.exists() else {}
        if result.get("status") != "ok" or result.get("file_sha256") != digest:
            if scorer is None:
                scorer = MultimodalScorer.from_checkpoints(args.checkpoint_dir, device, extractor, ffmpeg)
            try:
                result = dict(scorer.score(Path(row["file_path"])), status="ok", file_sha256=digest)
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                result = dict(status="error", error=str(exc), file_sha256=digest)
                tqdm.write(f"Skipped {row['source_file']}: {exc}")
            save_json(cache, result)
        results.append(dict(row, **result))
    results = pd.DataFrame(results)
    for split, table in results.groupby("split"):
        table.to_csv(args.output_dir / f"{split}_features.csv", index=False)
    print(results.groupby(["split", "category", "status"]).size().to_string())
    print(f"Saved: {args.output_dir}. Errors remain explicit; no invented features.")
    if not results["status"].eq("ok").any():
        raise RuntimeError("No clips scored successfully; inspect the reported errors")


if __name__ == "__main__":
    main()
