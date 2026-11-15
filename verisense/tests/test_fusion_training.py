"""Train-only scaling, validation-only selection, safe export and explicit rejects."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fusion.model import FEATURE_COLUMNS, LateFusionClassifier
from src.fusion.train_fusion import load_features, select_threshold, train_late_fusion


class FusionTrainingTests(unittest.TestCase):
    def fixtures(self, root):
        features, manifests = root / "features", root / "manifests"
        features.mkdir()
        manifests.mkdir()
        hashes = {}
        for split in ("train", "val", "test"):
            rows = []
            for group in range(3):
                for category, video, audio in (("real", False, False), ("video_only", True, False),
                                               ("audio_only", False, True), ("both", True, True)):
                    source = f"{split}/{group}_{category}.mp4"
                    row = dict(source_file=source, file_path=f"/fixture/{source}",
                               group_id=f"{split}/{group}", split=split, category=category,
                               modify_video=video, modify_audio=audio, label=int(video or audio))
                    rows.append(row)
            plan = pd.DataFrame(rows)
            plan.to_csv(manifests / f"{split}.csv", index=False)
            hashes[split] = hashlib.sha256((manifests / f"{split}.csv").read_bytes()).hexdigest()
            frame = plan.copy()
            for column in FEATURE_COLUMNS:
                flag = "modify_audio" if column.startswith("audio") else "modify_video"
                frame[column] = .2 + .5 * frame[flag].astype(int) + (.01 if split == "test" else 0)
            frame["status"], frame["face_frames"], frame["scored_audio_windows"] = "ok", 16, 3
            frame.loc[0, "status"] = "error"
            frame.loc[0, "error"] = "Insufficient face samples"
            frame.loc[0, list(FEATURE_COLUMNS)] = np.nan
            frame.to_csv(features / f"{split}_features.csv", index=False)
        (features / "run_metadata.json").write_text(json.dumps({
            "feature_columns": list(FEATURE_COLUMNS), "manifests": hashes,
            "checkpoints": dict.fromkeys(["image_head.pt", "video_best.pt", "audio_head.pt"], "fixture"),
        }))
        return features, manifests

    def test_cutoff_separates_close_scores_and_handles_ties(self):
        self.assertEqual(select_threshold([0, 0, 1, 1], [.45, .46, .48, .49]), .48)
        self.assertEqual(select_threshold([0, 1], [.48, .48]), .5)
        with self.assertRaises(ValueError):
            select_threshold([1, 1], [.2, .8])

    def test_training_export_matches_sklearn_and_does_not_use_test_to_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, manifests = self.fixtures(root)
            config = {"regularization_candidates": [.1, 1.0], "max_iter": 2000}
            with redirect_stdout(io.StringIO()):
                report = train_late_fusion(features, manifests, root / "run1", config)
            checkpoint = torch.load(root / "run1/fusion_best.pt", weights_only=True)
            tables, _, _ = load_features(features, manifests)
            scaler = StandardScaler().fit(tables["train"][list(FEATURE_COLUMNS)])
            classifier = LogisticRegression(C=checkpoint["C"], class_weight="balanced", max_iter=2000, random_state=42)
            classifier.fit(scaler.transform(tables["train"][list(FEATURE_COLUMNS)]), tables["train"]["label"])
            model = LateFusionClassifier(checkpoint)
            actual = model(tables["test"][list(FEATURE_COLUMNS)].to_numpy()).numpy()
            expected = classifier.predict_proba(scaler.transform(tables["test"][list(FEATURE_COLUMNS)]))[:, 1]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            self.assertEqual(report["coverage"]["test"]["rejected"], 1)
            self.assertEqual(sum(report["test_metrics"][key] for key in ("tn", "fp", "fn", "tp")), 11)
            test = pd.read_csv(features / "test_features.csv")
            test.loc[test["status"].eq("ok"), list(FEATURE_COLUMNS)] = .91
            test.to_csv(features / "test_features.csv", index=False)
            with redirect_stdout(io.StringIO()):
                train_late_fusion(features, manifests, root / "run2", config)
            changed = torch.load(root / "run2/fusion_best.pt", weights_only=True)
            for key in ("mean", "scale", "weight", "bias"):
                torch.testing.assert_close(checkpoint[key], changed[key], rtol=0, atol=0)
            self.assertEqual(checkpoint["decision_threshold"], changed["decision_threshold"])
            self.assertEqual(checkpoint["C"], changed["C"])
            for invalid in (np.full(5, np.nan), np.full(5, 1.1), np.zeros(4)):
                with self.assertRaises(ValueError):
                    model(invalid)
            with self.assertRaisesRegex(ValueError, "feature order"):
                LateFusionClassifier(dict(checkpoint, feature_columns=list(reversed(FEATURE_COLUMNS))))

    def test_rejects_partial_features_changed_manifests_and_invalid_success_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, manifests = self.fixtures(root)
            original = (features / "val_features.csv").read_bytes()
            frame = pd.read_csv(features / "val_features.csv")
            frame.iloc[:4].to_csv(features / "val_features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_features(features, manifests)
            (features / "val_features.csv").write_bytes(original)
            frame.loc[1, "audio_mean"] = np.nan
            frame.to_csv(features / "val_features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Unusable"):
                load_features(features, manifests)
            (features / "val_features.csv").write_bytes(original)
            with (manifests / "train.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                load_features(features, manifests)

    def test_rejects_cross_partition_families_even_when_the_clip_failed_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            features, manifests = self.fixtures(Path(directory))
            for path in (manifests / "test.csv", features / "test_features.csv"):
                frame = pd.read_csv(path)
                frame.loc[0, "group_id"] = "train/0"
                frame.to_csv(path, index=False)
            provenance = json.loads((features / "run_metadata.json").read_text())
            provenance["manifests"]["test"] = hashlib.sha256((manifests / "test.csv").read_bytes()).hexdigest()
            (features / "run_metadata.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, "cross partitions"):
                load_features(features, manifests)


if __name__ == "__main__":
    unittest.main()
