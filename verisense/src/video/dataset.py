"""
Video Branch Dataset: .npy temporal features or raw CLIP .npz with frame annotations.
Pads/truncates features and targets together; normalizes raw NPZ projections.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VideoEmbeddingDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        max_frames: int = 32,
        embedding_dim: int = 768,
        debug_limit: Optional[int] = None,
    ):
        self.csv_file = Path(csv_file)
        self.max_frames = max_frames
        self.embedding_dim = embedding_dim

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Video split CSV not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file, dtype={"video_id": str})
        self.source_groups = set(self.df["group_id"].dropna()) if "group_id" in self.df else set()
        if "status" in self.df:
            if not self.df["status"].isin(["ok", "error"]).all():
                raise ValueError("Unknown feature extraction status")
            usable = self.df["status"].eq("ok")
            print(f"{self.csv_file.name}: using {usable.sum()}/{len(self.df)} cached clips")
            self.df = self.df[usable].reset_index(drop=True)

        if self.df.empty or not self.df["label"].isin([0, 1]).all():
            raise ValueError("Video features require valid binary clip labels")
        if debug_limit is not None and debug_limit > 0:
            self.df = self.df.groupby("label", group_keys=False).head(max(1, debug_limit // 2)).head(debug_limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        npy_path = Path(row["embedding_path"])
        label = int(row["label"])
        video_id = str(row.get("video_id", row.get("source_file", npy_path.stem)))

        if npy_path.suffix == ".npz":
            with np.load(npy_path, allow_pickle=False) as data:
                features = data["features"].astype(np.float32)
                frame_labels = data["frame_labels"].copy()
                if (frame_labels.shape != (len(features),) or not np.isin(frame_labels, [0, 1]).all()
                        or int(data["video_label"]) != label or (label == 0 and frame_labels.any())):
                    raise ValueError(f"Invalid visual frame annotations: {npy_path}")
        else:
            features = np.load(npy_path, allow_pickle=False)
            frame_labels = np.full(len(features), label, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != self.embedding_dim or len(features) == 0:
            raise ValueError(f"Invalid video features: {npy_path}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite video features: {npy_path}")
        if npy_path.suffix == ".npz":
            norms = np.linalg.norm(features, axis=-1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 1e-12).any():
                raise ValueError(f"Zero-norm CLIP features: {npy_path}")
            features = features / norms

        t_len = features.shape[0]
        padded_labels = np.full(self.max_frames, -100, dtype=np.int64)

        # Truncate or pad
        if t_len >= self.max_frames:
            # Uniformly subsample indices
            indices = np.linspace(0, t_len - 1, self.max_frames, dtype=int)
            padded_features = features[indices]
            padded_labels[:] = frame_labels[indices]
            mask = np.ones(self.max_frames, dtype=bool)
            actual_length = self.max_frames
        else:
            padded_features = np.zeros((self.max_frames, self.embedding_dim), dtype=np.float32)
            padded_features[:t_len] = features
            padded_labels[:t_len] = frame_labels
            mask = np.zeros(self.max_frames, dtype=bool)
            mask[:t_len] = True
            actual_length = t_len

        return {
            "embeddings": torch.tensor(padded_features, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "actual_length": actual_length,
            "frame_labels": torch.tensor(padded_labels, dtype=torch.long),
        }
