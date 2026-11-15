"""Fusion coverage, preprocessing, paired scoring and source-split safeguards."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.preprocessing import prepare_audio
from src.fusion.build_features import (
    FEATURE_COLUMNS, MultimodalScorer, audio_windows, decode_audio, main, read_manifests,
)


class FusionFeatureTests(unittest.TestCase):
    def test_audio_windows_cover_whole_clip_including_tail(self):
        for seconds in (2.368, 4, 4.224, 7.616, 19.968):
            with self.subTest(seconds=seconds):
                wave = np.linspace(-1, 1, int(seconds * 16000), dtype=np.float32)
                windows = audio_windows(wave)
                starts = [round(start * 16000) for start, _ in windows]
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1], max(0, len(wave) - 64000))
                self.assertTrue(all(b - a <= 32000 for a, b in zip(starts, starts[1:])))
                for start, window in zip(starts, (w for _, w in windows)):
                    np.testing.assert_array_equal(window, prepare_audio(wave[start:start + 64000], 16000))

    def test_decoder_requires_real_nonempty_finite_bounded_audio(self):
        for result in (
            subprocess.CompletedProcess([], 1, b"", b"missing audio"),
            subprocess.CompletedProcess([], 0, b"", b""),
            subprocess.CompletedProcess([], 0, np.full(1600, np.nan, dtype="<f4").tobytes(), b""),
            subprocess.CompletedProcess([], 0, np.zeros(31 * 16000, dtype="<f4").tobytes(), b""),
        ):
            with patch("src.fusion.build_features.subprocess.run", return_value=result):
                with self.assertRaises(ValueError):
                    decode_audio(Path("paired.mp4"), "ffmpeg")

    def test_heads_receive_raw_vs_normalized_clip_features_and_all_audio_windows(self):
        raw = torch.tensor([[3.0, 4.0]] * 4)
        calls = {"audio": 0}

        def backbone(pixel_values):
            expected = -torch.tensor([.48145466, .4578275, .40821073]) / torch.tensor([.26862954, .26130258, .27577711])
            torch.testing.assert_close(pixel_values[0, :, 0, 0], expected)
            return SimpleNamespace(image_embeds=raw)

        def head(features):
            torch.testing.assert_close(features, raw)
            return torch.log(torch.tensor([[.8, .2]] * 4))

        def video(features, mask):
            torch.testing.assert_close(features, (raw / 5).unsqueeze(0))
            self.assertTrue(mask.all())
            return torch.log(torch.tensor([[.3, .7]])), None

        def audio(wave):
            calls["audio"] += 1
            self.assertEqual(tuple(wave.shape), (1, 64000))
            score = .2 * calls["audio"]
            return torch.log(torch.tensor([[1 - score, score]]))

        class Video:
            max_frames = 4
            __call__ = staticmethod(video)

        scorer = MultimodalScorer(
            SimpleNamespace(backbone=backbone, head=head), Video(), audio, torch.device("cpu"),
            SimpleNamespace(extract_faces=lambda rgb: [{"cropped_face": rgb}]), "ffmpeg",
        )
        frames = [(np.zeros((224, 224, 3), dtype=np.uint8), t) for t in range(4)]
        with patch("src.fusion.build_features.sample_video_frames", return_value=iter(frames)), \
                patch("src.fusion.build_features.decode_audio", return_value=np.ones(8 * 16000, dtype=np.float32)):
            scores = scorer.score(Path("same-paired-clip.mp4"))
        self.assertEqual(calls["audio"], 3)
        np.testing.assert_allclose([scores[key] for key in FEATURE_COLUMNS], [.2, .2, .7, .4, .6], rtol=1e-6)
        self.assertEqual(scores["audio_windows"], 3)

    def make_manifests(self, root):
        for split in ("train", "val", "test"):
            rows = []
            for label in (0, 1):
                clip = root / f"{split}_{label}.mp4"
                clip.write_bytes(b"fixture for manifest validation only")
                rows.append(dict(file_path=str(clip), source_file=f"{split}/{label}.mp4",
                                 group_id=f"{split}/original.mp4", label=label, split=split,
                                 category="both" if label else "real", modify_video=bool(label),
                                 modify_audio=bool(label), duration=8.0))
            pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)

    def test_manifest_rejects_source_overlap_and_wrong_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_manifests(root)
            self.assertEqual(len(read_manifests(root)), 6)
            val = pd.read_csv(root / "val.csv")
            val["group_id"] = "train/original.mp4"
            val.to_csv(root / "val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "cross partitions"):
                read_manifests(root)
            self.make_manifests(root)
            val = pd.read_csv(root / "val.csv")
            val["label"] = [1, 0]
            val.to_csv(root / "val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Labels disagree"):
                read_manifests(root)

    def test_resume_reuses_successful_clips_but_invalidates_changed_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_manifests(root)
            checkpoints, output = root / "weights", root / "output"
            checkpoints.mkdir()
            for name in ("image_head.pt", "video_best.pt", "audio_head.pt"):
                (checkpoints / name).write_bytes(b"checkpoint hash fixture")
            args = ["build_features", "--manifest-dir", str(root), "--checkpoint-dir", str(checkpoints),
                    "--output-dir", str(output)]
            scorer = SimpleNamespace(score=lambda path: dict.fromkeys(FEATURE_COLUMNS, .25))
            with patch.object(sys, "argv", args), \
                    patch("src.fusion.build_features.shutil.which", return_value="ffmpeg"), \
                    patch("src.fusion.build_features.FaceExtractor", return_value=SimpleNamespace(use_mediapipe=False)), \
                    patch.object(MultimodalScorer, "from_checkpoints", return_value=scorer) as loader:
                main()
                self.assertEqual(loader.call_count, 1)
                main()
                self.assertEqual(loader.call_count, 1)
                (root / "test_1.mp4").write_bytes(b"changed video")
                main()
                self.assertEqual(loader.call_count, 2)
                (checkpoints / "audio_head.pt").write_bytes(b"different model")
                with self.assertRaisesRegex(ValueError, "changed"):
                    main()
            self.assertEqual(len(list((output / "clips").glob("*.json"))), 6)
            self.assertEqual(json.loads((output / "run_metadata.json").read_text())["feature_columns"], list(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
