"""Cache frozen CLIP features for FaceForensics videos, reusing existing image splits."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.image.model import CLIPDeepfakeDetector
from src.video.face_extractor import FaceExtractor, sample_video_frames


def prepare_videos(video_dir, output_dir, split_dir, max_frames=16):
    video_files = sorted(Path(video_dir).rglob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(
            f"No MP4 videos found under {video_dir}. Colab raw videos are temporary; "
            "restore the FaceForensics++ downloads before loading CLIP."
        )
    output_dir = Path(output_dir)
    cache = output_dir / f"clip_l14_f{max_frames}_t30"
    cache.mkdir(parents=True, exist_ok=True)
    split_dir = Path(split_dir)
    source_splits = {}
    for split in ("train", "val", "test"):
        images = pd.read_csv(split_dir / f"image_{split}.csv", dtype={"video_id": str})
        for video_id in images.video_id:
            for source in video_id.split("_"):
                if source in source_splits and source_splits[source] != split:
                    raise ValueError(f"Source {source} overlaps existing splits")
                source_splits[source] = split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPDeepfakeDetector().to(device).eval()
    extractor = FaceExtractor()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    records = []
    skipped = []
    for path in tqdm(video_files, desc="Caching CLIP features"):
        parts = set(path.parts)
        if not parts.intersection({"original_sequences", "manipulated_sequences"}):
            continue
        split_names = {source_splits[s] for s in path.stem.split("_") if s in source_splits}
        if len(split_names) > 1:
            raise ValueError(f"Related sources overlap splits: {path.name}")
        if not split_names:
            skipped.append(path.name)
            continue
        for source in path.stem.split("_"):
            source_splits[source] = next(iter(split_names))
        label = int("manipulated_sequences" in parts)
        family = path.parent.parent.parent.name
        destination = cache / f"{family}_{path.stem}.npy"
        if not destination.exists():
            faces = []
            for rgb, timestamp in sample_video_frames(path, max_frames=max_frames):
                detected = extractor.extract_faces(rgb)
                if len(detected) == 1:
                    faces.append(detected[0]["cropped_face"])
            if not faces:
                skipped.append(path.name)
                continue
            pixels = torch.from_numpy(np.stack(faces)).permute(0, 3, 1, 2).float() / 255
            pixels = ((pixels - mean) / std).to(device)
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = model.extract_features(pixels).float().cpu().numpy()
            np.save(destination, features)
        records.append({"video_id": path.stem, "video_path": str(path.resolve()),
                        "embedding_path": str(destination.resolve()), "label": label,
                        "split": next(iter(split_names))})

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError("No usable videos found; check raw videos and the existing image splits")
    manifest.to_csv(output_dir / "video_manifest.csv", index=False)
    for name in ("train", "val", "test"):
        rows = manifest[manifest.split == name]
        if set(rows.label) != {0, 1}:
            raise ValueError(f"{name} requires both real and manipulated videos")
        rows.to_csv(split_dir / f"video_{name}.csv", index=False)
        print(name, rows.label.value_counts().to_dict())
    print(f"Cached {len(manifest)} videos; skipped {len(skipped)}. Skipped files: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-dir", required=True, help="Directory containing image_train/val/test.csv")
    parser.add_argument("--max-frames", type=int, default=16)
    args = parser.parse_args()
    prepare_videos(args.video_dir, args.output_dir, args.split_dir, args.max_frames)
