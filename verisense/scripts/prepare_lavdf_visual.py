"""Cache raw frozen CLIP features and timestamp labels for LAV-DF train/val only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video.face_extractor import FaceExtractor, sample_video_frames


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_labels(metadata, timestamps):
    """Use [start, end) seconds; audio-only edits never make a visual frame fake."""
    periods = metadata["fake_periods"]
    duration = metadata["duration"]
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid annotated duration")
    for period in periods:
        if (len(period) != 2 or not np.isfinite(period).all()
                or not 0 <= period[0] < period[1] <= duration + .1):
            raise ValueError(f"Invalid fake_periods in {metadata['file']}: {period}")
    if metadata["modify_video"] and not periods:
        raise ValueError(f"Missing visual edit periods: {metadata['file']}")
    return np.array([int(metadata["modify_video"] and any(start <= time < end for start, end in periods))
                     for time in timestamps], dtype=np.int64)


@torch.inference_mode()
def extract(path, metadata, encoder, extractor, device):
    crops, timestamps = [], []
    # Uniform sampling is independent of the annotation, including for validation.
    for rgb, time in sample_video_frames(path, max_frames=16):
        faces = extractor.extract_faces(rgb)
        if len(faces) == 1:
            crops.append(faces[0]["cropped_face"])
            timestamps.append(time)
    if len(crops) < 4:
        raise ValueError(f"Only {len(crops)} usable face frames; need four")
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None]
    features = []
    for start in range(0, len(crops), 4):
        pixels = torch.from_numpy(np.stack(crops[start:start + 4])).to(device)
        pixels = pixels.permute(0, 3, 1, 2).float() / 255
        features.append(encoder(pixel_values=(pixels - mean) / std).image_embeds.cpu().numpy())
    features = np.concatenate(features).astype(np.float32)
    if features.shape != (len(crops), 768) or not np.isfinite(features).all():
        raise ValueError("Invalid CLIP embeddings")
    return dict(features=features, timestamps=np.array(timestamps, dtype=np.float32),
                frame_labels=frame_labels(metadata, timestamps),
                video_label=np.array(int(metadata["modify_video"]), dtype=np.int64))


def prepare(subset_dir, checkpoint, output_dir):
    annotation_file = subset_dir / "selected_metadata.json"
    annotations = {row["file"]: row for row in json.loads(annotation_file.read_text())}
    plans = []
    for split in ("train", "val"):
        plan = pd.read_csv(subset_dir / "manifests" / f"{split}.csv", dtype={"group_id": str})
        if plan.empty or not plan["split"].eq(split).all():
            raise ValueError(f"Invalid {split} manifest")
        for row in plan.to_dict("records"):
            metadata = annotations[row["source_file"]]
            expected_split = "dev" if split == "val" else split
            if (metadata["split"] != expected_split
                    or (metadata["original"] or metadata["file"]) != row["group_id"]
                    or metadata["modify_video"] != row["modify_video"]
                    or metadata["modify_audio"] != row["modify_audio"]
                    or row["label"] != int(metadata["modify_video"] or metadata["modify_audio"])):
                raise ValueError("Annotations disagree with the saved source assignments")
            frame_labels(metadata, [])  # Validate intervals before downloading/loading CLIP.
            if not Path(row["file_path"]).is_file():
                raise FileNotFoundError(row["file_path"])
        plans.append(plan)
    plan = pd.concat(plans, ignore_index=True)
    if (plan["source_file"].duplicated().any() or plan["file_path"].duplicated().any()
            or plan.groupby("group_id")["split"].nunique().max() != 1):
        raise ValueError("Repeated clips or source families cross partitions")
    head = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if head["model_name"] != "openai/clip-vit-large-patch14" or head["image_size"] != 224:
        raise ValueError("Expected the existing CLIP L/14 image checkpoint")
    extractor = FaceExtractor()
    provenance = {
        "backbone": head["model_name"], "checkpoint_sha256": sha256(checkpoint),
        "annotation_sha256": sha256(annotation_file), "sampler": "16 uniform frames, first 30 seconds",
        "feature_format": "raw projected CLIP, not L2-normalized",
        "face_backend": "mediapipe" if extractor.use_mediapipe else "haar",
        "code_sha256": sha256(__file__),
        "face_code_sha256": sha256(Path(__file__).resolve().parents[1] / "src/video/face_extractor.py"),
        "manifests": {split: sha256(subset_dir / "manifests" / f"{split}.csv") for split in ("train", "val")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_file = output_dir / "provenance.json"
    if provenance_file.exists() and json.loads(provenance_file.read_text()) != provenance:
        raise ValueError("Inputs or extraction changed; choose a new cache directory")
    temporary = provenance_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(provenance, indent=2))
    temporary.replace(provenance_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Caching train/val visual features on {device}; test is reserved.", flush=True)
    encoder, results = None, []
    for row in tqdm(plan.to_dict("records"), desc="Caching visual embeddings"):
        key = hashlib.sha256(row["source_file"].encode()).hexdigest()
        cache = output_dir / "clips" / f"{key}.npz"
        record = cache.with_suffix(".json")
        digest = sha256(row["file_path"])
        result = json.loads(record.read_text()) if record.exists() else {}
        if result.get("file_sha256") != digest or (result.get("status") == "ok" and not cache.exists()):
            if encoder is None:
                encoder = CLIPVisionModelWithProjection.from_pretrained(head["model_name"]).to(device).eval()
                encoder.requires_grad_(False)
            try:
                arrays = extract(Path(row["file_path"]), annotations[row["source_file"]], encoder, extractor, device)
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(cache)
                result = dict(status="ok", embedding_path=str(cache), face_frames=len(arrays["frame_labels"]),
                              edited_frames=int(arrays["frame_labels"].sum()), file_sha256=digest)
            except (ValueError, OSError) as exc:
                result = dict(status="error", error=str(exc), file_sha256=digest)
                tqdm.write(f"Rejected {row['source_file']}: {exc}")
            record.parent.mkdir(parents=True, exist_ok=True)
            temporary = record.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, allow_nan=False))
            temporary.replace(record)
        results.append(dict(row, **result, overall_label=row["label"], label=int(row["modify_video"])))
    table = pd.DataFrame(results)
    for split, frame in table.groupby("split"):
        frame.to_csv(output_dir / f"{split}.csv", index=False)
        good = frame[frame["status"].eq("ok")]
        positive = int(good["edited_frames"].sum()) if not good.empty else 0
        total = int(good["face_frames"].sum()) if not good.empty else 0
        changed = good[good["modify_video"].eq(True)]
        covered = int((changed["edited_frames"] > 0).sum()) if len(changed) else 0
        print(f"{split}: usable clips={len(good)}/{len(frame)}, unmodified frames={total-positive}, edited frames={positive}")
        print(f"Visual-fake clips with a sampled edited face: {covered}/{len(changed)}")
    print(f"Saved raw features and timestamp labels to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.subset_dir, args.checkpoint, args.output_dir)
