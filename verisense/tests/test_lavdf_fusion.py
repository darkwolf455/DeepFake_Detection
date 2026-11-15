"""OOF group isolation, pooled alignment, resume and validation-only fusion."""
from contextlib import redirect_stdout
import importlib.util
import io
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("lavdf_oof", Path(__file__).resolve().parents[1] / "scripts/train_lavdf_fusion.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LavdfFusionTests(unittest.TestCase):
    def test_subset_keeps_ragged_labels_aligned_and_rejects_group_leakage(self):
        data = dict(features=torch.arange(5 * 768).reshape(5, 768).float(), labels=torch.tensor([0, 1, 0, 0, 1]),
                    offsets=np.array([0, 2, 5]), modality="visual",
                    clips=pd.DataFrame(dict(source_file=["a", "b"], group_id=["family_a", "family_b"])))
        selected = module.subset(data, np.array([False, True]))
        torch.testing.assert_close(selected["features"], data["features"][2:])
        self.assertEqual(selected["labels"].tolist(), [0, 0, 1])
        self.assertEqual(selected["offsets"].tolist(), [0, 3])
        with self.assertRaisesRegex(ValueError, "OOF training/held-out"):
            module.fit_fold({}, selected, selected, 1, torch.device("cpu"), 42)

    def fixtures(self, root):
        originals = root / "originals"
        originals.mkdir()
        torch.manual_seed(42)
        visual = nn.Sequential(nn.Dropout(0), nn.Linear(768, 2))
        with torch.no_grad():
            visual[-1].weight.zero_()
            visual[-1].bias.zero_()
        audio = nn.Sequential(nn.Linear(768, 8), nn.LayerNorm(8), nn.GELU(), nn.Dropout(0), nn.Linear(8, 2))
        with torch.no_grad():
            audio[-1].weight.zero_()
            audio[-1].bias.zero_()
        sources = {
            "visual": dict(model_name="openai/clip-vit-large-patch14", head_type="linear", hidden_dim=8,
                           dropout=0., image_size=224, class_names=["real", "manipulated"], head_state_dict=visual.state_dict()),
            "audio": dict(model_name="facebook/wav2vec2-base", hidden_dim=8, dropout=0., sample_rate=16000,
                          duration=4., class_names=["real", "manipulated"],
                          head_state_dict={**{f"head.{key}": value for key, value in audio.state_dict().items()},
                                           "layer_weights": torch.zeros(2), "pooler.fixture": torch.zeros(1)})}
        names = {"visual": "image_head.pt", "audio": "audio_head.pt"}
        final_paths = {"visual": root / "checkpoints/visual_lavdf_run1/image_lavdf_head.pt",
                       "audio": root / "checkpoints/audio_lavdf_run1/audio_lavdf_head.pt"}
        for modality, source in sources.items():
            torch.save(source, originals / names[modality])
        (originals / "video_best.pt").write_bytes(b"Fixture temporal-score identity; no inference from this fixture file")
        manifests, old = root / "fusion/lavdf_subset/manifests", root / "fusion/lavdf_features_v1"
        manifests.mkdir(parents=True)
        old.mkdir()
        for split in ("train", "val"):
            plan = []
            for group in range(3):
                for category, video, voice in (("real", 0, 0), ("video_only", 1, 0), ("audio_only", 0, 1), ("both", 1, 1)):
                    source = f"{split}/{group}_{category}.mp4"
                    plan.append(dict(source_file=source, group_id=f"{split}/family{group}", file_path=f"/fixture/{source}",
                                     split=split, category=category, modify_video=bool(video), modify_audio=bool(voice),
                                     label=int(video or voice), duration=8.))
            frame = pd.DataFrame(plan)
            frame.to_csv(manifests / f"{split}.csv", index=False)
            old_frame = frame.copy()
            for name in ("image_mean", "image_max", "audio_mean", "audio_max"):
                old_frame[name] = .5
            old_frame["video_score"] = .45 + .01 * frame["modify_video"].astype(int)
            old_frame["status"], old_frame["face_frames"], old_frame["scored_audio_windows"] = "ok", 4, 4
            old_frame.to_csv(old / f"{split}_features.csv", index=False)
            for modality in ("visual", "audio"):
                cache = root / f"adaptation/lavdf_{modality}_v1"
                cache.mkdir(parents=True, exist_ok=True)
                cached = frame.copy()
                flag = "modify_video" if modality == "visual" else "modify_audio"
                cached["label"] = cached[flag].astype(int)
                cached["status"] = "ok"
                for index, row in cached.iterrows():
                    labels = np.array([0, int(row[flag]), 0, int(row[flag])])
                    features = np.zeros((4, 768), dtype=np.float32)
                    features[:, 0] = labels * 2 - 1
                    path = cache / f"{split}_{index}.npz"
                    keys = dict(frame_labels=labels, video_label=np.array(row[flag])) if modality == "visual" else dict(window_labels=labels, audio_label=np.array(row[flag]))
                    np.savez(path, features=features, **keys)
                    cached.loc[index, "embedding_path"] = str(path)
                cached.to_csv(cache / f"{split}.csv", index=False)
        for modality in ("visual", "audio"):
            cache = root / f"adaptation/lavdf_{modality}_v1"
            provenance = dict(checkpoint_sha256=module.sha256(originals / names[modality]),
                              feature_format="raw projected CLIP, not L2-normalized" if modality == "visual" else "frozen Wav2Vec2 weighted attentive pooling")
            (cache / "provenance.json").write_text(json.dumps(provenance))
            datasets, _, _ = module.load_modality(cache, modality)
            head = module.make_head(sources[modality])
            for _ in module.training_epochs(head, datasets["train"]["features"], datasets["train"]["labels"], torch.device("cpu"), 1):
                pass
            adapted = {**sources[modality], "best_epoch": 1, "provenance": provenance}
            adapted["head_state_dict"] = ({**sources[modality]["head_state_dict"], **{f"head.{key}": value for key, value in head.state_dict().items()}}
                                           if modality == "audio" else head.state_dict())
            final_paths[modality].parent.mkdir(parents=True)
            torch.save(adapted, final_paths[modality])
        (old / "run_metadata.json").write_text(json.dumps({
            "checkpoints": {name: module.sha256(originals / name) for name in ("image_head.pt", "audio_head.pt", "video_best.pt")}}))
        (old / "test_features.csv").write_text("POISON: test scores must not be opened")
        (manifests / "test.csv").write_text("POISON: test labels must not be opened")
        return originals

    def test_oof_runs_without_test_data_and_resumes_existing_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = self.fixtures(root)
            with patch.object(module, "fit_fold", wraps=module.fit_fold) as fitter, redirect_stdout(io.StringIO()):
                report = module.rebuild(root, originals, root / "oof", root / "output")
                self.assertEqual(fitter.call_count, 6)
                module.rebuild(root, originals, root / "oof", root / "output")
                self.assertEqual(fitter.call_count, 6)
            self.assertNotIn("test_metrics", report)
            self.assertIn("val_metrics", report)
            assignments = pd.read_csv(root / "oof/oof_assignments.csv")
            self.assertEqual(assignments.groupby("group_id")["fold"].nunique().max(), 1)
            self.assertEqual(assignments["fold"].nunique(), 3)
            checkpoint = torch.load(root / "output/fusion_best.pt", weights_only=True)
            self.assertEqual(checkpoint["provenance"]["training_predictions"], "grouped_out_of_fold")
            self.assertEqual(checkpoint["provenance"]["folds"], 3)


if __name__ == "__main__":
    unittest.main()
