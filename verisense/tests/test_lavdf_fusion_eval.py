"""Frozen evaluation: no refit, unchanged cutoff, explicit rejections and resumable scores."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_lavdf_fusion as fixtures

spec = importlib.util.spec_from_file_location("lavdf_frozen_eval", Path(__file__).resolve().parents[1] / "scripts/evaluate_lavdf_fusion.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class FrozenFusionEvaluationTests(unittest.TestCase):
    def test_frozen_evaluation_resumes_rejects_drift_and_never_fits_on_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = fixtures.LavdfFusionTests().fixtures(root)
            code = Path(__file__).resolve().parents[1]
            for modality in ("visual", "audio"):
                cache = root / f"adaptation/lavdf_{modality}_v1/provenance.json"
                provenance = json.loads(cache.read_text())
                if modality == "visual":
                    provenance.update(face_backend="haar", face_code_sha256=evaluation.sha256(code / "src/video/face_extractor.py"))
                    adapted = root / "checkpoints/visual_lavdf_run1/image_lavdf_head.pt"
                else:
                    provenance.update(audio_code_sha256=evaluation.sha256(code / "src/audio/model.py"),
                                      preprocessing_sha256=evaluation.sha256(code / "src/audio/preprocessing.py"))
                    adapted = root / "checkpoints/audio_lavdf_run1/audio_lavdf_head.pt"
                cache.write_text(json.dumps(provenance))
                checkpoint = torch.load(adapted, weights_only=True)
                checkpoint["provenance"] = provenance
                torch.save(checkpoint, adapted)
            manifests = root / "fusion/lavdf_subset/manifests"
            old = root / "fusion/lavdf_features_v1"
            train = pd.read_csv(manifests / "train.csv")
            test = train.copy()
            for name in ("source_file", "group_id"):
                test[name] = test[name].str.replace("train/", "test/", regex=False)
            test["split"] = "test"
            test.to_csv(manifests / "test.csv", index=False)
            for split in ("train", "val", "test"):
                plan = pd.read_csv(manifests / f"{split}.csv")
                for index, row in plan.iterrows():
                    clip = root / row["source_file"]
                    clip.parent.mkdir(exist_ok=True)
                    clip.write_bytes(row["source_file"].encode())
                    plan.loc[index, "file_path"] = str(clip)
                plan.to_csv(manifests / f"{split}.csv", index=False)
                if split != "test":
                    scores = pd.read_csv(old / f"{split}_features.csv")
                    scores["file_path"] = plan["file_path"]
                    scores.to_csv(old / f"{split}_features.csv", index=False)
            run = root / "checkpoints/fusion_run2"
            with redirect_stdout(io.StringIO()):
                fixtures.module.rebuild(root, originals, root / "oof", run)
            frozen = (run / "fusion_best.pt").read_bytes()
            expected_cutoff = torch.load(run / "fusion_best.pt", weights_only=True)["decision_threshold"]
            def score(path):
                if path.name == "0_real.mp4":
                    raise ValueError("Insufficient face frames")
                fake = "real" not in path.name
                return dict.fromkeys(evaluation.FEATURE_COLUMNS, .8 if fake else .2) | dict(face_frames=4, scored_audio_windows=2)
            with patch.object(evaluation, "FaceExtractor", return_value=SimpleNamespace(use_mediapipe=False)), \
                    patch.object(evaluation.shutil, "which", return_value="ffmpeg"), \
                    patch.object(evaluation.MultimodalScorer, "from_checkpoints", return_value=SimpleNamespace(score=score)) as loader, \
                    patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("No test fitting")), \
                    patch("src.fusion.train_fusion.select_threshold", side_effect=AssertionError("No test cutoff selection")), \
                    redirect_stdout(io.StringIO()):
                report = evaluation.evaluate(root, originals, root / "test_cache", root / "test_report")
                evaluation.evaluate(root, originals, root / "test_cache", root / "test_report")
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(report["coverage"]["test"]["rejected"], 1)
                self.assertEqual(report["coverage"]["test"]["scored"], 11)
                self.assertEqual(report["decision_threshold"], expected_cutoff)
                self.assertEqual((run / "fusion_best.pt").read_bytes(), frozen)
                predictions = pd.read_csv(root / "test_report/fusion_test_predictions.csv")
                expected = (predictions["fusion_score"] >= expected_cutoff).astype(int)
                self.assertEqual(predictions["fusion_prediction"].tolist(), expected.tolist())
                saved = json.loads((root / "test_report/fusion_test_metrics.json").read_text())
                self.assertEqual(saved["test_metrics"], report["test_metrics"])
                modified = root / "checkpoints/audio_lavdf_run1/audio_lavdf_head.pt"
                modified.write_bytes(modified.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "differs from the weights"):
                    evaluation.evaluate(root, originals, root / "test_cache", root / "test_report")
                # Family checks include the deliberately rejected real clip.
                plan = pd.read_csv(manifests / "test.csv")
                plan.loc[0, "group_id"] = "train/family0"
                plan.to_csv(manifests / "test.csv", index=False)
                with self.assertRaisesRegex(ValueError, "cross partitions"):
                    evaluation.evaluate(root, originals, root / "test_cache", root / "test_report")


if __name__ == "__main__":
    unittest.main()
