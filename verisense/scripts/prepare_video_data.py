"""
VeriSense: Prepare Video Data
Extracts faces from video clips at uniform intervals, runs frozen CLIP encoder,
and caches per-frame feature embeddings (.npy) and face boxes to disk for rapid training.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Ensure parent directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.video.face_extractor import FaceExtractor
import open_clip


def extract_video_features(
    video_path: Path,
    extractor: FaceExtractor,
    clip_model,
    clip_preprocess,
    device: torch.device,
    fps_sample: float = 2.0,
    max_frames: int = 32,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, False

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 25.0

    frame_interval = max(1, int(video_fps / fps_sample))
    frame_idx = 0
    sampled_faces = []
    has_detected_face = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = extractor.extract_faces(rgb_frame)
            if len(faces) > 0:
                has_detected_face = True
                # Select primary face
                primary_face = faces[0]["cropped_face"]
                sampled_faces.append(primary_face)
            else:
                # If no face, center crop as fallback
                h, w, _ = rgb_frame.shape
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                center_crop = cv2.resize(rgb_frame[top:top+min_dim, left:left+min_dim], (224, 224))
                sampled_faces.append(center_crop)

            if len(sampled_faces) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(sampled_faces) == 0:
        return None, 0, False

    # Pad or subsample to max_frames
    pil_images = [Image.fromarray(f) for f in sampled_faces]
    batch_tensors = torch.stack([clip_preprocess(img) for img in pil_images]).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            features = clip_model.encode_image(batch_tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()

    return features, len(sampled_faces), has_detected_face


def main():
    parser = argparse.ArgumentParser(description="Extract faces and cache CLIP embeddings for videos")
    parser.add_argument("--video-dir", type=str, required=True, help="Path to raw video directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/video_cache", help="Cache directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames sampled per second")
    parser.add_argument("--max-frames", type=int, default=32, help="Max frames per video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    embeddings_dir = output_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP ViT-L/14 on {device}...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    clip_model.eval()

    face_extractor = FaceExtractor(margin_ratio=0.25, target_size=(224, 224))

    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        video_files.extend(list(Path(args.video_dir).rglob(ext)))

    print(f"Found {len(video_files)} video clips to process.")
    records = []

    for vpath in tqdm(video_files):
        video_id = vpath.stem
        is_fake = any(s in str(vpath).lower() for s in ["fake", "manipulated", "deepfake", "face2face", "faceswap"])
        label = 1 if is_fake else 0

        npy_path = embeddings_dir / f"{video_id}.npy"

        if npy_path.exists():
            features = np.load(npy_path)
            records.append({
                "video_id": video_id,
                "video_path": str(vpath.resolve()),
                "embedding_path": str(npy_path.resolve()),
                "num_frames": len(features),
                "label": label,
                "face_detected": True
            })
            continue

        features, num_frames, face_detected = extract_video_features(
            vpath, face_extractor, clip_model, clip_preprocess, device, args.fps, args.max_frames
        )

        if features is not None:
            np.save(npy_path, features)
            records.append({
                "video_id": video_id,
                "video_path": str(vpath.resolve()),
                "embedding_path": str(npy_path.resolve()),
                "num_frames": num_frames,
                "label": label,
                "face_detected": face_detected
            })

    manifest_path = output_path / "video_manifest.csv"
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)
    print(f"Extraction complete! Manifest written to {manifest_path} with {len(df)} records.")


if __name__ == "__main__":
    main()
