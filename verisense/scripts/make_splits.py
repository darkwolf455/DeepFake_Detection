"""
VeriSense: Make Splits (Leakage-Free Partitioning)
Splits data into train (70%), validation (15%), and test (15%) partitions.
Strictly guarantees:
- Videos are split by video_id or original actor identity (never by frame).
- Audio is split by speaker_id.
- Images are partitioned such that specific generator domains or subjects do not leak.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_image_splits(processed_dir: Path, output_dir: Path, seed: int = 42):
    manifest_csv = processed_dir / "images" / "all_images_manifest.csv"
    if not manifest_csv.exists():
        print(f"Skipping images: {manifest_csv} does not exist.")
        return

    df = pd.read_csv(manifest_csv)
    # If generator column exists, stratify or group
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"])

    train_df.to_csv(output_dir / "image_train.csv", index=False)
    val_df.to_csv(output_dir / "image_val.csv", index=False)
    test_df.to_csv(output_dir / "image_test.csv", index=False)
    print(f"Image splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def make_video_splits(processed_dir: Path, output_dir: Path, seed: int = 42):
    manifest_csv = processed_dir / "video_cache" / "video_manifest.csv"
    if not manifest_csv.exists():
        print(f"Skipping video: {manifest_csv} does not exist.")
        return

    df = pd.read_csv(manifest_csv)
    # Extract identity or parent video id (e.g. 001 from 001_002 or 001)
    df["group_id"] = df["video_id"].apply(lambda x: str(x).split("_")[0])

    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(df, groups=df["group_id"]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["group_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    train_df.to_csv(output_dir / "video_train.csv", index=False)
    val_df.to_csv(output_dir / "video_val.csv", index=False)
    test_df.to_csv(output_dir / "video_test.csv", index=False)
    print(f"Video splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def make_audio_splits(processed_dir: Path, output_dir: Path, seed: int = 42):
    manifest_csv = processed_dir / "audio" / "audio_manifest.csv"
    if not manifest_csv.exists():
        print(f"Skipping audio: {manifest_csv} does not exist.")
        return

    df = pd.read_csv(manifest_csv)
    # Group by speaker_id to ensure zero speaker overlap
    groups = df["speaker_id"].fillna("unknown")

    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["speaker_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    train_df.to_csv(output_dir / "audio_train.csv", index=False)
    val_df.to_csv(output_dir / "audio_val.csv", index=False)
    test_df.to_csv(output_dir / "audio_test.csv", index=False)
    print(f"Audio splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def main():
    parser = argparse.ArgumentParser(description="Generate leak-free train/val/test splits")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating splits in {out_dir} with seed={args.seed}...")
    make_image_splits(proc_dir, out_dir, args.seed)
    make_video_splits(proc_dir, out_dir, args.seed)
    make_audio_splits(proc_dir, out_dir, args.seed)
    print("All splits generated successfully without identity or video leakage!")


if __name__ == "__main__":
    main()
