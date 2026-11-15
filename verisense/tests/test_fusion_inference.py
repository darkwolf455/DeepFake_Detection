"""Frozen head loading, shared backbones, explicit fusion routing and abstentions."""
import asyncio
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.fusion import load_fusion
from backend.inference import InferenceManager
from src.fusion.build_features import sha256
from src.fusion.model import FEATURE_COLUMNS, LateFusionClassifier


def classifier_checkpoint():
    return dict(model_type="standardized_logistic_regression", feature_columns=list(FEATURE_COLUMNS),
                class_names=["real", "manipulated"], threshold_source="validation", decision_threshold=.5,
                mean=torch.zeros(5), scale=torch.ones(5), weight=torch.tensor([1., 2., -1., -2., 1.]), bias=torch.tensor([0.]))


class SpeechFixture(nn.Module):
    backbone_name = "speech-fixture"

    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.layer_weights = nn.Parameter(torch.ones(1))
        self.pooler = nn.Linear(768, 8)
        self.head = nn.Linear(8, 2)

    def extract_features(self, inputs):
        return self.pooler(inputs[:, :768])


class FusionInferenceTests(unittest.TestCase):
    def test_loader_shares_backbones_preserves_original_heads_and_rejects_changed_weights(self):
        image = SimpleNamespace(model_name="visual-fixture", backbone=nn.Identity(), head=nn.Linear(768, 2))
        audio = SpeechFixture().eval()
        before = copy.deepcopy(audio.state_dict()), copy.deepcopy(image.head.state_dict())
        extractor = SimpleNamespace(use_mediapipe=False)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = {name: folder / name for name in ("image_head.pt", "video_best.pt", "audio_head.pt")}
            for path in source.values():
                torch.save({"fixture": True}, path)
            adapted = folder / "lavdf"
            adapted.mkdir()
            visual_state = copy.deepcopy(image.head.state_dict())
            visual_state["weight"] += 1
            speech_state = {key: value.clone() for key, value in audio.state_dict().items()}
            speech_state["head.weight"] += 1
            torch.save(dict(model_name=image.model_name, image_size=224, class_names=["real", "manipulated"],
                            head_state_dict=visual_state), adapted / "image_lavdf_head.pt")
            torch.save(dict(model_name=audio.backbone_name, sample_rate=16000, duration=4.,
                            class_names=["real", "manipulated"], head_state_dict=speech_state), adapted / "audio_lavdf_head.pt")
            (adapted / "video_best.pt").write_bytes(source["video_best.pt"].read_bytes())
            lineage = dict(training_predictions="grouped_out_of_fold",
                source_checkpoints={name: sha256(path) for name, path in source.items() if name != "video_best.pt"},
                checkpoints={"image_head.pt": sha256(adapted / "image_lavdf_head.pt"),
                    "audio_head.pt": sha256(adapted / "audio_lavdf_head.pt"), "video_best.pt": sha256(source["video_best.pt"])},
                caches=dict(visual=dict(face_backend="haar", face_code_sha256=sha256(root / "src/video/face_extractor.py")),
                    audio=dict(audio_code_sha256=sha256(root / "src/audio/model.py"),
                               preprocessing_sha256=sha256(root / "src/audio/preprocessing.py"))))
            checkpoint = classifier_checkpoint() | dict(provenance=lineage)
            path = adapted / "fusion_best.pt"
            torch.save(checkpoint, path)
            classifier, scorer = load_fusion(path, source, image, SimpleNamespace(max_frames=16), audio,
                                             torch.device("cpu"), extractor)
            self.assertIs(scorer.image.backbone, image.backbone)
            self.assertIs(scorer.audio.source, audio)
            inputs = torch.randn(2, 64000)
            expected = nn.Linear(8, 2)
            expected.load_state_dict({key.removeprefix("head."): value for key, value in speech_state.items() if key.startswith("head.")})
            torch.testing.assert_close(scorer.audio(inputs), expected(audio.extract_features(inputs)))
            for name, value in audio.state_dict().items():
                torch.testing.assert_close(value, before[0][name], rtol=0, atol=0)
            for name, value in image.head.state_dict().items():
                torch.testing.assert_close(value, before[1][name], rtol=0, atol=0)
            self.assertEqual(classifier.decision_threshold, .5)
            self.assertTrue(Path(scorer.ffmpeg).is_file())
            speech_state["pooler.bias"] += 1
            speech = torch.load(adapted / "audio_lavdf_head.pt", weights_only=True)
            speech["head_state_dict"] = speech_state
            torch.save(speech, adapted / "audio_lavdf_head.pt")
            with self.assertRaisesRegex(ValueError, "differ from the evaluated"):
                load_fusion(path, source, image, SimpleNamespace(max_frames=16), audio, torch.device("cpu"), extractor)
            lineage["checkpoints"]["audio_head.pt"] = sha256(adapted / "audio_lavdf_head.pt")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "pooling changed"):
                load_fusion(path, source, image, SimpleNamespace(max_frames=16), audio, torch.device("cpu"), extractor)

    def test_explicit_fusion_uses_all_five_inputs_its_cutoff_and_no_lipsync(self):
        manager = InferenceManager.__new__(InferenceManager)
        manager.semaphore = asyncio.Semaphore(1)
        manager.fusion_model = LateFusionClassifier(classifier_checkpoint())
        values = dict(zip(FEATURE_COLUMNS, [.1, .8, .4, .2, .9]), face_frames=8, scored_audio_windows=3)
        manager.fusion_scorer = SimpleNamespace(score=lambda path: values)
        expected = float(manager.fusion_model([values[name] for name in FEATURE_COLUMNS]))
        result = asyncio.run(manager.analyze_file(Path("clip.mp4"), "video", "hash", use_fusion=True))
        self.assertEqual(result.scores.fusion, round(expected, 4))
        self.assertEqual(result.verdict, "manipulated")
        self.assertEqual((result.scores.image, result.scores.audio), (.8, .9))
        self.assertIsNone(result.scores.av_sync)
        self.assertIsNone(result.timeline)
        self.assertEqual({model.modality for model in result.models}, {"image", "video", "audio", "fusion"})
        self.assertTrue(any("cutoff of 0.500000" in warning for warning in result.warnings))
        self.assertFalse(any("FaceForensics++" in warning for warning in result.warnings))
        manager._analyze_video = Mock(side_effect=AssertionError("No silent fallback"))
        manager.fusion_scorer.score = Mock(side_effect=ValueError("Missing soundtrack"))
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(manager.analyze_file(Path("silent.mp4"), "video", "hash", use_fusion=True))
        self.assertEqual(caught.exception.status_code, 422)
        manager._analyze_video.assert_not_called()


if __name__ == "__main__":
    unittest.main()
