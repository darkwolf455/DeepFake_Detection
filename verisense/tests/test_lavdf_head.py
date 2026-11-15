"""Frozen-feature adaptation, honest candidate selection and checkpoint preservation."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("train_lavdf_head", Path(__file__).resolve().parents[1] / "scripts/train_lavdf_head.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LavdfHeadTests(unittest.TestCase):
    def fixtures(self, root, perfect=False):
        checkpoint = root / "image_head.pt"
        head = nn.Sequential(nn.Dropout(0), nn.Linear(768, 2))
        with torch.no_grad():
            head[1].weight.zero_()
            head[1].bias.zero_()
            if perfect:
                head[1].weight[0, 0], head[1].weight[1, 0] = -2, 2
        torch.save(dict(model_name="openai/clip-vit-large-patch14", head_type="linear", hidden_dim=8,
                        dropout=0., image_size=224, class_names=["real", "manipulated"],
                        head_state_dict=head.state_dict()), checkpoint)
        cache = root / "cache"
        cache.mkdir()
        for split in ("train", "val"):
            rows = []
            for index in range(4):
                label = index % 2
                labels = np.array([0, 1, 0, 0, 0, 1, 0, 0] if label else [0] * 8)
                features = np.zeros((8, 768), dtype=np.float32)
                features[:, 0] = labels * 2 - 1
                path = cache / f"{split}_{index}.npz"
                np.savez(path, features=features, frame_labels=labels, video_label=np.array(label))
                rows.append(dict(source_file=f"{split}/{index}.mp4", file_path=f"/fixture/{split}/{index}.mp4",
                                 group_id=f"{split}/{index}", split=split, modify_video=bool(label), label=label,
                                 status="ok", embedding_path=str(path)))
            pd.DataFrame(rows).to_csv(cache / f"{split}.csv", index=False)
        (cache / "test.csv").write_text("POISON: held-out test must not be opened by the trainer")
        (cache / "provenance.json").write_text(json.dumps({
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "feature_format": "raw projected CLIP, not L2-normalized"}))
        return cache, checkpoint

    def test_adaptation_exports_real_head_without_reading_test_or_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, checkpoint = self.fixtures(root)
            original = checkpoint.read_bytes()
            with redirect_stdout(io.StringIO()):
                report = module.train(cache, checkpoint, root / "output", epochs=2)
            self.assertTrue(report["improved"])
            self.assertGreater(report["selected_validation"]["clip_auc"], report["baseline_validation"]["clip_auc"])
            self.assertEqual(original, checkpoint.read_bytes())
            export = torch.load(root / "output/image_lavdf_head.pt", weights_only=True)
            self.assertEqual(export["threshold_scope"], "max_over_16_face_samples")
            head = module.make_head(export).eval()
            data, _ = module.load_cache(cache)
            validation, scores = module.evaluate(head, data["val"], torch.device("cpu"))
            self.assertEqual(validation, report["selected_validation"])
            saved = pd.read_csv(root / "output/visual_val_predictions.csv")
            np.testing.assert_allclose(saved["adapted_score"], scores)

    def test_keeps_original_when_validation_auc_does_not_improve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, checkpoint = self.fixtures(root, perfect=True)
            with redirect_stdout(io.StringIO()):
                report = module.train(cache, checkpoint, root / "output", epochs=2)
            self.assertFalse(report["improved"])
            self.assertEqual(report["best_epoch"], 0)
            original = torch.load(checkpoint, weights_only=True)["head_state_dict"]
            selected = torch.load(root / "output/image_lavdf_head.pt", weights_only=True)["head_state_dict"]
            for key in original:
                torch.testing.assert_close(original[key], selected[key], rtol=0, atol=0)

    def test_rejects_wrong_modality_labels_and_overlapping_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, _ = self.fixtures(root)
            frame = pd.read_csv(cache / "val.csv")
            original = frame.copy()
            frame.loc[0, "label"] = 1  # Audio-only/real visual frame cannot inherit the overall fake label.
            frame.to_csv(cache / "val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Invalid visual labels"):
                module.load_cache(cache)
            original.loc[0, "group_id"] = "train/0"
            original.to_csv(cache / "val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "cross train/validation"):
                module.load_cache(cache)


if __name__ == "__main__":
    unittest.main()
