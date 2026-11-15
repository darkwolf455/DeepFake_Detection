"""Audio preprocessing, compact checkpoints and API routing without network downloads."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.inference import InferenceManager
from src.audio.model import Wav2Vec2DeepfakeDetector
from src.audio.preprocessing import prepare_audio


class BackboneFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8))
        self.config = SimpleNamespace(hidden_size=8, num_hidden_layers=1)

    def forward(self, inputs, output_hidden_states):
        features = inputs[:, ::320, None] * self.weight
        return SimpleNamespace(hidden_states=(features, features * 0.5))


class AudioInferenceTests(unittest.TestCase):
    def setUp(self):
        self.waveform = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32)
        self.manager = InferenceManager.__new__(InferenceManager)
        self.manager.device = torch.device("cpu")
        self.manager.audio_threshold = 0.505932629108429
        self.manager.audio_model = lambda inputs: torch.log(torch.tensor([[0.48, 0.52]]))
        self.manager.image_model = None
        self.manager.semaphore = asyncio.Semaphore(1)

    def decoder(self, waveform=None, rate=16000, frames=None):
        waveform = self.waveform if waveform is None else waveform
        decoder = SimpleNamespace(samplerate=rate, channels=1,
                                  frames=len(waveform) if frames is None else frames,
                                  read=lambda **kwargs: waveform)
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = decoder
        return patch.dict(sys.modules, {"soundfile": SimpleNamespace(SoundFile=lambda path: context)})

    def test_center_crop_resampling_and_repeat_padding(self):
        rate = 8000
        # Extra tones outside the central four seconds must not be analyzed.
        frequencies = np.repeat([110, 220, 330], [rate, 4 * rate, rate])
        source = np.sin(2 * np.pi * np.cumsum(frequencies) / rate).astype(np.float32)
        result = prepare_audio(np.column_stack([source, source]), rate)
        self.assertEqual(result.shape, (64000,))
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(np.sqrt(np.mean(result ** 2))), 0.1, places=5)
        peak = np.fft.rfftfreq(len(result), 1 / 16000)[np.argmax(abs(np.fft.rfft(result)))]
        self.assertEqual(peak, 220)
        short = prepare_audio(self.waveform, 16000)
        np.testing.assert_array_equal(short[:16000], short[16000:32000])
        for bad in [[], [np.nan], [np.inf], np.zeros((2, 2, 2))]:
            with self.assertRaises(ValueError):
                prepare_audio(bad, 16000)

    def test_audio_routes_without_an_image_model_and_uses_its_cutoff(self):
        with self.decoder():
            result = asyncio.run(self.manager.analyze_file(Path("speech.wav"), "audio", "hash"))
        self.assertEqual(result.verdict, "manipulated")
        self.assertAlmostEqual(result.scores.audio, 0.52)
        self.assertEqual(result.models[0].name, "Wav2Vec2 Base Speech Classifier")
        self.assertTrue(any("ASVspoof" in warning for warning in result.warnings))
        self.assertIsNone(result.scores.image)
        self.assertIsNone(result.timeline)

    def test_invalid_silent_and_oversized_audio_are_rejected(self):
        for samples, frames, expected in [(np.array([]), 0, 400), (np.array([np.nan]), 1, 400),
                                           (np.zeros(16000), 16000, 422), (self.waveform, 31 * 16000, 413)]:
            with self.subTest(status=expected), self.decoder(samples, frames=frames):
                with self.assertRaises(HTTPException) as caught:
                    self.manager._analyze_audio(Path("speech.wav"), "hash")
                self.assertEqual(caught.exception.status_code, expected)
        with self.decoder():
            self.manager.audio_model = lambda inputs: torch.tensor([[float("nan"), 0]])
            with self.assertRaises(HTTPException) as caught:
                self.manager._analyze_audio(Path("speech.wav"), "hash")
            self.assertEqual(caught.exception.status_code, 500)

    def test_compact_checkpoint_restores_only_the_trained_head(self):
        with patch("src.audio.model.AutoModel.from_pretrained", side_effect=lambda *args, **kwargs: BackboneFixture()):
            original = Wav2Vec2DeepfakeDetector(hidden_dim=16, dropout=0).eval()
            checkpoint = dict(model_name="facebook/wav2vec2-base", class_names=["real", "manipulated"],
                              sample_rate=16000, duration=4.0, hidden_dim=16, dropout=0,
                              decision_threshold=0.505932629108429, threshold_source="validation",
                              head_state_dict={k: v for k, v in original.state_dict().items() if not k.startswith("backbone.")})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "audio_head.pt"
                torch.save(checkpoint, path)
                restored = Wav2Vec2DeepfakeDetector.from_checkpoint(torch.load(path, weights_only=True)).eval()
            inputs = torch.from_numpy(self.waveform).unsqueeze(0)
            torch.testing.assert_close(original(inputs), restored(inputs))
            self.assertTrue(all(not p.requires_grad for p in restored.backbone.parameters()))
            for field, invalid in [("decision_threshold", float("nan")), ("decision_threshold", 1.0),
                                   ("threshold_source", "test"), ("sample_rate", 44100),
                                   ("class_names", ["manipulated", "real"]), ("head_state_dict", {})]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    Wav2Vec2DeepfakeDetector.from_checkpoint({**checkpoint, field: invalid})


if __name__ == "__main__":
    unittest.main()
