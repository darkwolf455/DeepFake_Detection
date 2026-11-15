"""Video routing/aggregation checks with real decoding and controlled face scores."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.inference import InferenceManager


class VideoBaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video = Path(self.directory.name) / "clip.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (224, 224))
        self.assertTrue(writer.isOpened())
        for _ in range(160):  # 40 seconds: verify sampling ends before 30 seconds.
            writer.write(np.zeros((224, 224, 3), dtype=np.uint8))
        writer.release()
        self.manager = InferenceManager.__new__(InferenceManager)
        self.manager.face_extractor = SimpleNamespace(extract_faces=lambda rgb: [
            {"cropped_face": rgb}
        ])

    def test_averages_real_sample_scores_and_bounds_the_timeline(self):
        scores = iter(np.linspace(0.1, 0.9, 16))
        self.manager._score_face = lambda crop: float(next(scores))
        result = self.manager._analyze_video(self.video, "test-hash")
        self.assertAlmostEqual(result.scores.video, 0.5)
        self.assertEqual(result.verdict, "inconclusive")
        self.assertEqual(len(result.timeline), 16)
        times = [point.t for point in result.timeline]
        self.assertEqual(times, sorted(set(times)))
        self.assertLess(times[-1], 30)
        self.assertIsNone(result.scores.audio)
        self.assertIsNone(result.scores.av_sync)
        self.assertIn("no temporal", result.warnings[1])

    def test_no_usable_faces_returns_validation_error(self):
        for count in (0, 2):
            with self.subTest(faces=count):
                self.manager.face_extractor.extract_faces = lambda rgb: [
                    {"cropped_face": rgb} for _ in range(count)
                ]
                with self.assertRaises(HTTPException) as caught:
                    self.manager._analyze_video(self.video, "hash")
                self.assertEqual(caught.exception.status_code, 422)

    def test_corrupt_video_returns_decode_error(self):
        broken = Path(self.directory.name) / "broken.mp4"
        broken.write_bytes(b"invalid video")
        with self.assertRaises(HTTPException) as caught:
            self.manager._analyze_video(broken, "hash")
        self.assertEqual(caught.exception.status_code, 400)

    def test_temporal_mode_uses_sequence_score_instead_of_frame_mean(self):
        class SequenceFixture:
            max_frames = 4

            def __call__(self, features, mask):
                return torch.tensor([[0.0, 2.0]]), torch.tensor([[[2.0, 0.0]] * features.shape[1]])

        self.manager.device = torch.device("cpu")
        self.manager.video_model = SequenceFixture()
        self.manager.image_model = SimpleNamespace(extract_features=lambda pixels: torch.zeros(len(pixels), 768))
        self.manager._face_tensor = lambda crop: torch.zeros(1, 3, 224, 224)
        result = self.manager._analyze_video(self.video, "hash")
        self.assertAlmostEqual(result.scores.video, 0.8808)
        self.assertEqual(result.verdict, "manipulated")
        self.assertEqual(len(result.timeline), 4)
        self.assertLess(result.timeline[0].fake_prob, 0.2)
        self.assertIn("Temporal", result.models[0].name)

    def test_validation_cutoff_applies_only_to_temporal_video(self):
        class SequenceFixture:
            max_frames = 4

            def __call__(self, features, mask):
                logits = torch.log(torch.tensor([[0.51, 0.49]]))
                return logits, torch.zeros(1, features.shape[1], 2)

        self.manager.device = torch.device("cpu")
        self.manager.video_model = SequenceFixture()
        self.manager.video_threshold = 0.48329490423202515
        self.manager.image_model = SimpleNamespace(extract_features=lambda pixels: torch.zeros(len(pixels), 768))
        self.manager._face_tensor = lambda crop: torch.zeros(1, 3, 224, 224)
        result = self.manager._analyze_video(self.video, "hash")
        self.assertEqual(result.verdict, "manipulated")
        self.assertAlmostEqual(result.scores.video, 0.49)
        self.assertAlmostEqual(result.confidence, 0.49)
        self.assertTrue(any("validation-selected cutoff" in warning for warning in result.warnings))
        for score, expected in [(0.48, "authentic"), (self.manager.video_threshold, "manipulated")]:
            result = self.manager._response(score, "video", "hash", time.perf_counter(),
                                            decision_threshold=self.manager.video_threshold)
            self.assertEqual(result.verdict, expected)
        image = self.manager._response(0.49, "image", "hash", time.perf_counter())
        self.assertEqual(image.verdict, "inconclusive")


if __name__ == "__main__":
    unittest.main()
