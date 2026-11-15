"""Haar regression checks; portrait cases use the user's local problem_face.png."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video.face_extractor import FaceExtractor

PORTRAIT = Path(__file__).resolve().parents[2] / "problem_face.png"


class FaceExtractorTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(sys.modules, {"mediapipe": None}):
            self.extractor = FaceExtractor()

    def count(self, bgr):
        return len(self.extractor.extract_faces(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def test_blank_image_has_no_faces(self):
        self.assertEqual(self.count(np.zeros((494, 426, 3), dtype=np.uint8)), 0)

    @unittest.skipUnless(PORTRAIT.exists(), "Local portrait fixture not present")
    def test_single_face_at_different_sizes_and_compression(self):
        image = cv2.imread(str(PORTRAIT))
        _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 30])
        cases = {
            "original": image,
            "half_size": cv2.resize(image, None, fx=0.5, fy=0.5),
            "compressed": cv2.imdecode(jpeg, cv2.IMREAD_COLOR),
        }
        for name, sample in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.count(sample), 1)

    @unittest.skipUnless(PORTRAIT.exists(), "Local portrait fixture not present")
    def test_two_faces_are_not_collapsed_to_one(self):
        image = cv2.imread(str(PORTRAIT))
        self.assertEqual(self.count(np.concatenate([image, image], axis=1)), 2)


if __name__ == "__main__":
    unittest.main()
