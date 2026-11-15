"""Annotated frame alignment, weighted supervision and safe temporal warm starts."""
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video.dataset import VideoEmbeddingDataset
from src.video.model import VideoTemporalDetector
from src.video.train import train_video_epoch, train_video_model


class LavdfTemporalTests(unittest.TestCase):
    def test_npz_normalization_and_frame_targets_follow_padding_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "features.npz"
            features = np.zeros((5, 8), dtype=np.float32)
            features[:, :2] = [3, 4]
            labels = np.array([0, 1, 0, 1, 0])
            np.savez(path, features=features, frame_labels=labels, video_label=np.array(1))
            csv = root / "clips.csv"
            pd.DataFrame([dict(source_file="train/001.mp4", embedding_path=str(path), label=1)]).to_csv(csv, index=False)
            padded = VideoEmbeddingDataset(csv, max_frames=8, embedding_dim=8)[0]
            np.testing.assert_allclose(padded["embeddings"][:5, :2], np.tile([.6, .8], (5, 1)), atol=1e-7)
            self.assertEqual(padded["frame_labels"].tolist(), [0, 1, 0, 1, 0, -100, -100, -100])
            self.assertEqual(padded["video_id"], "train/001.mp4")
            truncated = VideoEmbeddingDataset(csv, max_frames=3, embedding_dim=8)[0]
            self.assertEqual(truncated["frame_labels"].tolist(), labels[[0, 2, 4]].tolist())
            np.savez(path, features=features, frame_labels=labels, video_label=np.array(0))
            with self.assertRaisesRegex(ValueError, "annotations"):
                VideoEmbeddingDataset(csv, max_frames=8, embedding_dim=8)[0]

    def test_frame_loss_receives_local_labels_and_excludes_padding(self):
        torch.manual_seed(42)
        model = VideoTemporalDetector(input_dim=8, d_model=16, nhead=2, num_layers=1,
                                      dim_feedforward=32, max_frames=4, dropout=0)
        batch = dict(embeddings=torch.randn(2, 4, 8), label=torch.tensor([1, 0]),
                     mask=torch.tensor([[True, True, True, False], [True, True, False, False]]),
                     frame_labels=torch.tensor([[0, 1, 0, -100], [0, 0, -100, -100]]))
        received = []
        class FrameLoss(nn.Module):
            def forward(self, logits, labels):
                received.append(labels.detach().clone())
                return nn.functional.cross_entropy(logits, labels, weight=torch.tensor([.5, 2.]))
        train_video_epoch(model, [batch], torch.optim.AdamW(model.parameters(), lr=.001),
                          nn.CrossEntropyLoss(), torch.amp.GradScaler("cuda", enabled=False),
                          torch.device("cpu"), frame_criterion=FrameLoss())
        self.assertEqual(received[0].tolist(), [0, 1, 0, 0, 0])

    def test_unobserved_positive_edit_is_not_used_as_positive_clip_training_target(self):
        model = VideoTemporalDetector(input_dim=8, d_model=16, nhead=2, num_layers=1,
                                      dim_feedforward=32, max_frames=4, dropout=0)
        batch = dict(embeddings=torch.randn(2, 4, 8), label=torch.tensor([1, 0]),
                     mask=torch.ones(2, 4, dtype=torch.bool), frame_labels=torch.zeros(2, 4, dtype=torch.long))
        received = []
        class ClipLoss(nn.Module):
            def forward(self, logits, labels):
                received.append(labels.tolist())
                return nn.functional.cross_entropy(logits, labels)
        train_video_epoch(model, [batch], torch.optim.AdamW(model.parameters(), lr=.001),
                          ClipLoss(), torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
                          frame_criterion=nn.CrossEntropyLoss(), annotated_clip_loss=True)
        self.assertEqual(received, [[0]])

    def test_warm_start_is_retained_when_validation_does_not_improve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporal = dict(type="transformer", input_dim=8, d_model=16, nhead=2, num_layers=1,
                            dim_feedforward=32, max_frames=4, dropout=0.)
            model = VideoTemporalDetector.from_config(temporal)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()  # A constant baseline; no feature-dependent gradient can improve AUC.
            warm = root / "original.pt"
            torch.save(dict(config={"temporal_model": temporal}, model_state_dict=model.state_dict()), warm)
            original = warm.read_bytes()
            for split in ("train", "val"):
                rows = []
                for index in range(4):
                    label = index % 2
                    path = root / f"{split}_{index}.npz"
                    np.savez(path, features=np.ones((4, 8), dtype=np.float32),
                             frame_labels=np.array([0, label, label if split == "val" else 0, 0]),
                             video_label=np.array(label))
                    rows.append(dict(source_file=f"{split}/{index}.mp4", group_id=f"{split}/{index}",
                                     embedding_path=str(path), label=label, status="ok"))
                pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)
            config = dict(temporal_model=temporal,
                          data=dict(train_split_csv=str(root / "train.csv"), val_split_csv=str(root / "val.csv"),
                                    test_split_csv="POISON_DO_NOT_READ", num_workers=0, batch_size=2, val_batch_size=4,
                                    preload_embeddings=True),
                          training=dict(seed=42, lr=.0001, weight_decay=.005, epochs=1, use_amp=False,
                                        early_stopping_patience=4, mode="max", checkpoint_dir=str(root / "output"),
                                        checkpoint_name="video_lavdf_best.pt", frame_supervision="annotated",
                                        frame_class_weights=[.6, 4.], gradient_accumulation_steps=1))
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            with redirect_stdout(io.StringIO()):
                report = train_video_model(path, warm_start=warm)
            self.assertEqual(report["best_epoch"], 0)
            self.assertEqual(warm.read_bytes(), original)
            selected = torch.load(root / "output/video_lavdf_best.pt", weights_only=True)
            self.assertEqual(selected["threshold_source"], "validation")
            self.assertEqual(selected["frame_supervision"], "weak")
            self.assertEqual(selected["attempted_frame_supervision"], "annotated")
            np.testing.assert_allclose(selected["config"]["training"]["frame_class_weights"], [16/28, 4.])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, selected["model_state_dict"][key], rtol=0, atol=0)
            self.assertEqual(len(pd.read_csv(root / "output/video_val_predictions.csv")), 4)
            validation = pd.read_csv(root / "val.csv")
            validation.loc[0, "group_id"] = "train/0"
            validation.to_csv(root / "val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "cross training/validation"):
                train_video_model(path, warm_start=warm)

    def test_adaptation_wrapper_keeps_test_and_validation_weights_out_of_config(self):
        spec = importlib.util.spec_from_file_location("temporal_adaptation", Path(__file__).resolve().parents[1] / "scripts/train_lavdf_temporal.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warm = root / "original.pt"
            torch.save(dict(config=dict(temporal_model={"max_frames": 16},
                                        data={"test_split_csv": "POISON"}, training={})), warm)
            (root / "provenance.json").write_text(json.dumps({"fixture": True}))
            with patch.object(module, "train_video_model", return_value={}) as trainer, redirect_stdout(io.StringIO()):
                module.adapt(root, warm, root / "output")
            config_path = trainer.call_args.args[0]
            config = yaml.safe_load(config_path.read_text())
            self.assertNotIn("frame_class_weights", config["training"])
            self.assertEqual(config["training"]["frame_supervision"], "annotated")
            self.assertNotIn("test_split_csv", config["data"])
            self.assertTrue(config["data"]["preload_embeddings"])


if __name__ == "__main__":
    unittest.main()
