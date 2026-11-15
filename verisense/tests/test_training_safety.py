"""Regression checks for real feature inputs, portable checkpoints, and frame supervision."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import ConfigDict
from src.common.utils import load_checkpoint, save_checkpoint
from src.video.dataset import VideoEmbeddingDataset
from src.video.model import AttentionPooling, VideoTemporalDetector
from src.video.train import train_video_epoch


class TrainingSafetyTests(unittest.TestCase):
    def test_float16_pooling_ignores_padding_and_has_finite_gradients(self):
        torch.manual_seed(42)
        pooler = AttentionPooling(8).half()
        valid = torch.randn(2, 2, 8, dtype=torch.float16)
        padded = torch.cat([valid, torch.full_like(valid, 1000)], dim=1).requires_grad_()
        mask = torch.tensor([[True, True, False, False]]).expand(2, -1)
        pooled = pooler(padded, mask)
        torch.testing.assert_close(pooled, pooler(valid))
        pooled.float().square().mean().backward()
        self.assertTrue(torch.isfinite(padded.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in pooler.parameters()))
        self.assertEqual(padded.grad[:, 2:].count_nonzero().item(), 0)

    def test_checkpoint_with_config_loads_with_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(4, 2)
            config = ConfigDict({"temporal_model": {"d_model": 128}})
            save_checkpoint({"model_state_dict": model.state_dict(), "config": config}, directory)
            restored = torch.nn.Linear(4, 2)
            checkpoint = load_checkpoint(Path(directory) / "checkpoint_best.pt", restored)
            self.assertIs(type(checkpoint["config"]), dict)
            self.assertIs(type(checkpoint["config"]["temporal_model"]), dict)
            torch.testing.assert_close(model.weight, restored.weight)

    def test_missing_empty_and_nonfinite_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "features.npy"
            csv = root / "videos.csv"
            pd.DataFrame([{"video_id": "001", "embedding_path": str(path), "label": 0}]).to_csv(csv, index=False)
            dataset = VideoEmbeddingDataset(csv, embedding_dim=8)
            with self.assertRaises(FileNotFoundError):
                dataset[0]
            for data in [np.zeros((0, 8)), np.zeros((2, 4)), np.full((2, 8), np.nan)]:
                np.save(path, data)
                with self.assertRaises(ValueError):
                    dataset[0]
            np.save(path, np.ones((2, 8), dtype=np.float32))
            self.assertEqual(dataset[0]["video_id"], "001")

    def test_training_updates_the_frame_classifier(self):
        torch.manual_seed(42)
        model = VideoTemporalDetector(input_dim=8, d_model=16, nhead=2, num_layers=1,
                                      dim_feedforward=32, max_frames=4, dropout=0)
        batch = {"embeddings": torch.randn(2, 4, 8),
                 "mask": torch.tensor([[True, True, True, False], [True, True, False, False]]),
                 "label": torch.tensor([0, 1])}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.frame_classifier[-1].weight.detach().clone()
        train_video_epoch(model, [batch], optimizer, torch.nn.CrossEntropyLoss(),
                          torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"))
        self.assertFalse(torch.equal(before, model.frame_classifier[-1].weight))


if __name__ == "__main__":
    unittest.main()
