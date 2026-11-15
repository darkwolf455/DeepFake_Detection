"""Audio interval targets, resume behavior and complete frozen-pooler exports."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("adapt_audio", Path(__file__).resolve().parents[1] / "scripts/adapt_lavdf_audio.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from src.audio.model import Wav2Vec2DeepfakeDetector
from train_lavdf_head import train, make_head


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768, num_hidden_layers=1)


class LavdfAudioTests(unittest.TestCase):
    def test_audio_windows_overlap_only_audio_edits_with_half_open_boundaries(self):
        metadata = dict(file="train/clip.mp4", duration=8., fake_periods=[[1., 2.]], modify_audio=True)
        bounds = [(0, 1), (1, 2), (2, 4), (1.5, 4)]
        np.testing.assert_array_equal(module.window_labels(metadata, bounds), [0, 1, 0, 1])
        np.testing.assert_array_equal(module.window_labels(dict(metadata, modify_audio=False), bounds), [0] * 4)
        with self.assertRaises(ValueError):
            module.window_labels(dict(metadata, fake_periods=[]), bounds)

    def test_visual_only_manipulation_keeps_audio_label_zero_and_success_cache_resumes(self):
        class Encoder(nn.Module):
            def extract_features(self, tensor):
                self_shape = tuple(tensor.shape)
                if self_shape != (1, 64000):
                    raise AssertionError(self_shape)
                return torch.ones(1, 768)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            annotations = []
            for split, audio in [("train", True), ("val", False)]:
                source = f"{'dev' if split == 'val' else split}/clip.mp4"
                path = root / f"{split}.mp4"
                path.write_bytes(split.encode())
                row = dict(source_file=source, file_path=str(path), group_id=source, split=split,
                           modify_video=True, modify_audio=audio, label=1, category="both" if audio else "video_only")
                pd.DataFrame([row]).to_csv(root / "manifests" / f"{split}.csv", index=False)
                annotations.append(dict(file=source, original=None, split="dev" if split == "val" else split,
                                        modify_video=True, modify_audio=audio, duration=8., fake_periods=[[2., 3.]]))
            (root / "selected_metadata.json").write_text(json.dumps(annotations))
            checkpoint = root / "source.pt"
            torch.save({"fixture": True}, checkpoint)
            with patch.object(module.shutil, "which", return_value="ffmpeg"), \
                    patch.object(module, "decode_audio", return_value=np.ones(8 * 16000, dtype=np.float32)), \
                    patch.object(module.Wav2Vec2DeepfakeDetector, "from_checkpoint", return_value=Encoder()) as loader, \
                    redirect_stdout(io.StringIO()):
                module.prepare(root, checkpoint, root / "cache")
                module.prepare(root, checkpoint, root / "cache")
                self.assertEqual(loader.call_count, 1)
            val = pd.read_csv(root / "cache/val.csv")
            self.assertEqual(val.loc[0, "label"], 0)
            self.assertEqual(val.loc[0, "overall_label"], 1)
            with np.load(val.loc[0, "embedding_path"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["window_labels"].tolist(), [0, 0, 0])
            train_table = pd.read_csv(root / "cache/train.csv")
            with np.load(train_table.loc[0, "embedding_path"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["window_labels"].tolist(), [1, 1, 0])

    def test_audio_export_retains_pooling_weights_and_distinguishes_window_and_clip_cutoffs(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.audio.model.AutoModel.from_pretrained", return_value=Backbone()):
            root = Path(directory)
            original = Wav2Vec2DeepfakeDetector(hidden_dim=8, dropout=0).eval()
            with torch.no_grad():
                original.head[-1].weight.zero_()
                original.head[-1].bias.zero_()
            source = dict(model_name="facebook/wav2vec2-base", sample_rate=16000, duration=4.,
                          class_names=["real", "manipulated"], hidden_dim=8, dropout=0.,
                          decision_threshold=.5, threshold_source="validation",
                          head_state_dict={key: value for key, value in original.state_dict().items() if not key.startswith("backbone.")})
            checkpoint = root / "audio_head.pt"
            torch.save(source, checkpoint)
            cache = root / "cache"
            cache.mkdir()
            for split in ("train", "val"):
                rows = []
                for index in range(4):
                    label = index % 2
                    labels = np.array([0, label, 0, label])
                    features = np.zeros((4, 768), dtype=np.float32)
                    features[:, 0] = labels * 2 - 1
                    path = cache / f"{split}_{index}.npz"
                    np.savez(path, features=features, window_labels=labels, audio_label=np.array(label))
                    rows.append(dict(source_file=f"{split}/{index}.mp4", file_path=f"/fixture/{split}/{index}.mp4",
                                     group_id=f"{split}/{index}", split=split, modify_audio=bool(label), label=label,
                                     status="ok", embedding_path=str(path)))
                pd.DataFrame(rows).to_csv(cache / f"{split}.csv", index=False)
            (cache / "test.csv").write_text("POISON: do not read test data")
            (cache / "provenance.json").write_text(json.dumps({
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "feature_format": "frozen Wav2Vec2 weighted attentive pooling"}))
            with redirect_stdout(io.StringIO()):
                report = train(cache, checkpoint, root / "output", epochs=2, modality="audio")
            export = torch.load(root / "output/audio_lavdf_head.pt", weights_only=True)
            self.assertEqual(export["threshold_scope"], "4_second_window")
            self.assertEqual(export["decision_threshold"], report["window_decision_threshold"])
            self.assertEqual(export["clip_decision_threshold"], report["clip_decision_threshold"])
            for key in source["head_state_dict"]:
                if not key.startswith("head."):
                    torch.testing.assert_close(source["head_state_dict"][key], export["head_state_dict"][key], rtol=0, atol=0)
            restored = Wav2Vec2DeepfakeDetector.from_checkpoint(export).eval()
            features = torch.randn(3, 768)
            torch.testing.assert_close(restored.head(features), make_head(export).eval()(features))


if __name__ == "__main__":
    unittest.main()
