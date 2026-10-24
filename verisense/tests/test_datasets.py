"""
Unit tests for Dataset loading, sequence padding, and attention masks.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.video.dataset import VideoEmbeddingDataset
from src.video.model import VideoTemporalDetector


def test_video_embedding_dataset_padding(tmp_path):
    # Create synthetic test csv and npy
    csv_file = tmp_path / "video_test.csv"
    npy_file = tmp_path / "test_vid.npy"

    # 10 frames of 768-dim features
    features = np.random.randn(10, 768).astype(np.float32)
    np.save(npy_file, features)

    df = pd.DataFrame([{"video_id": "test_vid", "embedding_path": str(npy_file), "label": 1}])
    df.to_csv(csv_file, index=False)

    ds = VideoEmbeddingDataset(csv_file, max_frames=32, embedding_dim=768)
    sample = ds[0]

    assert sample["embeddings"].shape == (32, 768)
    assert sample["mask"].shape == (32,)
    assert sample["mask"][:10].all()
    assert not sample["mask"][10:].any()
    assert sample["label"].item() == 1


def test_video_temporal_model_forward():
    model = VideoTemporalDetector(input_dim=768, d_model=256, nhead=4, num_layers=2, max_frames=32)
    model.eval()

    dummy_input = torch.randn(2, 32, 768)
    dummy_mask = torch.ones(2, 32, dtype=torch.bool)
    dummy_mask[:, 20:] = False

    video_logits, frame_logits = model(dummy_input, mask=dummy_mask)

    assert video_logits.shape == (2, 2)
    assert frame_logits.shape == (2, 32, 2)
