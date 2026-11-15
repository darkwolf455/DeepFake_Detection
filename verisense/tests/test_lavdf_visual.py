"""Localized visual labels and raw CLIP feature caching."""
import importlib.util
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("prepare_lavdf_visual", Path(__file__).resolve().parents[1] / "scripts/prepare_lavdf_visual.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LavdfVisualTests(unittest.TestCase):
    def metadata(self, modify_video=True):
        return dict(file="train/example.mp4", duration=8., fake_periods=[[1., 2.], [5., 6.]], modify_video=modify_video)

    def test_only_edited_times_are_visual_positives(self):
        times = [0., 1., 1.99, 2., 4., 5.5, 6.]
        np.testing.assert_array_equal(module.frame_labels(self.metadata(), times), [0, 1, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(module.frame_labels(self.metadata(False), times), np.zeros(len(times)))

    def test_invalid_or_missing_intervals_fail(self):
        for periods in ([], [[2., 1.]], [[-1., 2.]], [[1., 20.]], [[1., np.nan]]):
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                module.frame_labels(dict(self.metadata(), fake_periods=periods), [1.5])

    def test_sampling_is_label_independent_and_features_remain_unnormalized(self):
        class Encoder:
            def __call__(self, pixel_values):
                expected = -torch.tensor([.48145466, .4578275, .40821073]) / torch.tensor([.26862954, .26130258, .27577711])
                torch.testing.assert_close(pixel_values[0, :, 0, 0], expected)
                return SimpleNamespace(image_embeds=torch.full((len(pixel_values), 768), 3.))
        frames = [(np.zeros((224, 224, 3), dtype=np.uint8), time) for time in [0., 1.5, 3., 5.5]]
        detector = SimpleNamespace(extract_faces=lambda rgb: [{"cropped_face": rgb}])
        with patch.object(module, "sample_video_frames", return_value=iter(frames)) as sampler:
            result = module.extract(Path("paired.mp4"), self.metadata(), Encoder(), detector, torch.device("cpu"))
        sampler.assert_called_once_with(Path("paired.mp4"), max_frames=16)
        np.testing.assert_array_equal(result["frame_labels"], [0, 1, 0, 1])
        self.assertTrue((result["features"] == 3).all())
        self.assertEqual(result["features"].shape, (4, 768))

    def test_audio_only_clip_has_negative_visual_manifest_label_and_cache_resumes(self):
        class Encoder(torch.nn.Module):
            def forward(self, pixel_values):
                return SimpleNamespace(image_embeds=torch.ones(len(pixel_values), 768))

        frames = [(np.zeros((224, 224, 3), dtype=np.uint8), time) for time in [0., 1.5, 3., 5.5]]
        detector = SimpleNamespace(use_mediapipe=False, extract_faces=lambda rgb: [{"cropped_face": rgb}])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifests").mkdir()
            annotations = []
            for split, video, audio in [("train", True, False), ("val", False, True)]:
                source = f"{'dev' if split == 'val' else split}/example.mp4"
                clip = root / f"{split}.mp4"
                clip.write_bytes(split.encode())
                row = dict(source_file=source, file_path=str(clip), group_id=source, split=split,
                           label=1, modify_video=video, modify_audio=audio)
                pd.DataFrame([row]).to_csv(root / "manifests" / f"{split}.csv", index=False)
                annotations.append(dict(self.metadata(video), file=source, original=None,
                                        split="dev" if split == "val" else split, modify_audio=audio))
            (root / "selected_metadata.json").write_text(json.dumps(annotations))
            checkpoint = root / "image_head.pt"
            torch.save(dict(model_name="openai/clip-vit-large-patch14", image_size=224), checkpoint)
            with patch.object(module, "FaceExtractor", return_value=detector), \
                    patch.object(module, "sample_video_frames", side_effect=lambda *a, **kw: iter(frames)), \
                    patch.object(module.CLIPVisionModelWithProjection, "from_pretrained", return_value=Encoder()) as loader:
                module.prepare(root, checkpoint, root / "cache")
                module.prepare(root, checkpoint, root / "cache")
                self.assertEqual(loader.call_count, 1)
            val = pd.read_csv(root / "cache/val.csv")
            self.assertEqual(val.loc[0, "label"], 0)
            self.assertEqual(val.loc[0, "overall_label"], 1)
            with np.load(val.loc[0, "embedding_path"], allow_pickle=False) as arrays:
                self.assertEqual(int(arrays["video_label"]), 0)
                self.assertFalse(arrays["frame_labels"].any())


if __name__ == "__main__":
    unittest.main()
