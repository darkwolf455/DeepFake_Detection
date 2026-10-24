"""
Video Branch Dataset: Reads precomputed CLIP frame embeddings (.npy) or video clips.
Pads or truncates sequences to max_frames with attention masks.
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

        self.df = pd.read_csv(self.csv_file)

        if debug_limit is not None and debug_limit > 0:
            self.df = self.df.head(debug_limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        npy_path = Path(row["embedding_path"])
        label = int(row["label"])
        video_id = str(row["video_id"])

        if npy_path.exists():
            features = np.load(npy_path)  # Shape: (T, embed_dim)
        else:
            # Synthetic fallback if cache file missing during test/smoke
            features = np.zeros((self.max_frames, self.embedding_dim), dtype=np.float32)

        t_len = features.shape[0]

        # Truncate or pad
        if t_len >= self.max_frames:
            # Uniformly subsample indices
            indices = np.linspace(0, t_len - 1, self.max_frames, dtype=int)
            padded_features = features[indices]
            mask = np.ones(self.max_frames, dtype=bool)
            actual_length = self.max_frames
        else:
            padded_features = np.zeros((self.max_frames, self.embedding_dim), dtype=np.float32)
            padded_features[:t_len] = features
            mask = np.zeros(self.max_frames, dtype=bool)
            mask[:t_len] = True
            actual_length = t_len

        return {
            "embeddings": torch.tensor(padded_features, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "actual_length": actual_length,
        }
