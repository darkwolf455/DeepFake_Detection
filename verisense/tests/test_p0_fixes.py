"""Focused tests for the P0 image-pipeline fixes.

These tests deliberately avoid downloading CLIP weights or training anything.
Run from the ``verisense`` directory with:

    .venv/bin/python -m unittest tests.test_p0_fixes -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import albumentations as A
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from backend.schemas import image_model_version
from src.common.config import ConfigDict, load_config
from src.common.utils import (
    get_device,
    load_checkpoint,
    load_checkpoint_dict,
    save_checkpoint,
    set_seed,
)
from src.image.dataset import (
    DeepfakeImageDataset,
    build_jpeg_compression,
    get_image_transforms,
)
from src.image.evaluate import get_robustness_transform
from src.image.model import CLIPDeepfakeDetector
from src.image.train import build_train_loader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "image.yaml"


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


class DeviceAndSeedTests(unittest.TestCase):
    def test_cuda_is_preferred(self):
        with mock.patch("src.common.utils.torch.cuda.is_available", return_value=True), \
             mock.patch("src.common.utils.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(get_device().type, "cuda")

    def test_mps_is_preferred_without_cuda(self):
        with mock.patch("src.common.utils.torch.cuda.is_available", return_value=False), \
             mock.patch("src.common.utils.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(get_device().type, "mps")

    def test_cpu_is_the_final_fallback(self):
        with mock.patch("src.common.utils.torch.cuda.is_available", return_value=False), \
             mock.patch("src.common.utils.torch.backends.mps.is_available", return_value=False):
            self.assertEqual(get_device().type, "cpu")

    def test_explicit_mps_falls_back_to_cpu(self):
        with mock.patch("src.common.utils.torch.backends.mps.is_available", return_value=False):
            self.assertEqual(get_device("mps").type, "cpu")

    def test_set_seed_uses_mps(self):
        with mock.patch("src.common.utils.torch.cuda.is_available", return_value=False), \
             mock.patch("src.common.utils.torch.backends.mps.is_available", return_value=True), \
             mock.patch("src.common.utils.torch.manual_seed") as manual_seed, \
             mock.patch("src.common.utils.torch.mps.manual_seed") as mps_seed:
            set_seed(123)
        manual_seed.assert_called_once_with(123)
        mps_seed.assert_called_once_with(123)


class CheckpointTests(unittest.TestCase):
    def test_configdict_to_dict_is_plain(self):
        cfg = ConfigDict({"model": {"name": "x"}, "items": [ConfigDict({"a": 1})]})
        plain = cfg.to_dict()
        self.assertIs(type(plain), dict)
        self.assertIs(type(plain["model"]), dict)
        self.assertIs(type(plain["items"][0]), dict)

    def test_plain_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            cfg = load_config(CONFIG_PATH)
            save_checkpoint(
                {
                    "epoch": 1,
                    "model_state_dict": {"w": torch.ones(1)},
                    "config": cfg.to_dict(),
                },
                tmp,
                filename="ckpt.pt",
            )

            loaded = load_checkpoint_dict(path)
            self.assertIs(type(loaded["config"]), dict)
            self.assertNotIsInstance(loaded["config"], ConfigDict)

    def test_legacy_configdict_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            cfg = load_config(CONFIG_PATH)
            torch.save({"model_state_dict": {"w": torch.ones(1)}, "config": cfg}, path)

            loaded = load_checkpoint_dict(path)
            self.assertIsInstance(loaded["config"], ConfigDict)

    def test_load_checkpoint_into_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint({"model_state_dict": {"w": torch.ones(1)}}, tmp, filename="model.pt")

            model = _TinyModel()
            load_checkpoint(path, model)
            self.assertTrue(torch.equal(model.w, torch.ones(1)))


class TransformTests(unittest.TestCase):
    def test_training_jpeg_uses_configured_quality_range(self):
        cfg = load_config(CONFIG_PATH)
        transform = get_image_transforms(cfg, is_training=True)
        compression = next(t for t in transform.transforms if isinstance(t, A.ImageCompression))
        self.assertEqual(tuple(compression.quality_range), (30, 100))

    def test_robustness_jpeg_uses_requested_quality(self):
        transform = get_robustness_transform(224, jpeg_quality=30)
        compression = next(t for t in transform.transforms if isinstance(t, A.ImageCompression))
        self.assertEqual(tuple(compression.quality_range), (30, 30))

    def test_jpeg_helper_returns_compression_transform(self):
        self.assertIsInstance(build_jpeg_compression(30, 100, p=1.0), A.ImageCompression)


class DatasetTests(unittest.TestCase):
    @staticmethod
    def _write_split(tmp, image_path):
        csv_path = Path(tmp) / "split.csv"
        pd.DataFrame(
            [{"file_path": str(image_path), "label": 1, "generator": "test"}]
        ).to_csv(csv_path, index=False)
        return csv_path

    def test_reads_valid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "real.png"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(img_path)

            ds = DeepfakeImageDataset(self._write_split(tmp, img_path), transform=None)
            sample = ds[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))

    def test_missing_image_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "missing.jpg"
            ds = DeepfakeImageDataset(self._write_split(tmp, img_path), transform=None)

            with self.assertRaisesRegex(RuntimeError, "Failed to read image"):
                _ = ds[0]


class TrainLoaderTests(unittest.TestCase):
    def test_drop_last_removes_trailing_single_sample(self):
        loader = build_train_loader(
            TensorDataset(torch.zeros(3, 1)), batch_size=2, num_workers=0
        )
        self.assertEqual([len(batch[0]) for batch in loader], [2])

    def test_smaller_dataset_stays_in_one_batch(self):
        loader = build_train_loader(
            TensorDataset(torch.zeros(3, 1)), batch_size=8, num_workers=0
        )
        self.assertEqual([len(batch[0]) for batch in loader], [3])

    def test_batch_size_one_is_rejected(self):
        with self.assertRaises(ValueError):
            build_train_loader(TensorDataset(torch.zeros(3, 1)), batch_size=1)

    def test_single_image_split_is_rejected(self):
        with self.assertRaises(ValueError):
            build_train_loader(TensorDataset(torch.zeros(1, 1)), batch_size=8)


class ModelConfigTests(unittest.TestCase):
    def test_from_config_maps_saved_architecture(self):
        saved = {
            "model": {
                "name": "custom/clip",
                "freeze_backbone": False,
                "head_type": "linear",
                "hidden_dim": 256,
                "dropout": 0.1,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            }
        }

        with mock.patch.object(CLIPDeepfakeDetector, "__init__", return_value=None) as init:
            CLIPDeepfakeDetector.from_config(saved)

        kwargs = init.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "custom/clip")
        self.assertFalse(kwargs["freeze_backbone"])
        self.assertEqual(kwargs["head_type"], "linear")
        self.assertEqual(kwargs["hidden_dim"], 256)
        self.assertEqual(kwargs["dropout"], 0.1)
        self.assertTrue(kwargs["use_lora"])
        self.assertEqual(kwargs["lora_r"], 16)
        self.assertEqual(kwargs["lora_alpha"], 32)
        self.assertEqual(kwargs["lora_dropout"], 0.05)

    def test_from_checkpoint_returns_none_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(CLIPDeepfakeDetector, "from_config") as from_config:
                model = CLIPDeepfakeDetector.from_checkpoint(Path(tmp) / "missing.pt")

        self.assertIsNone(model)
        from_config.assert_not_called()

    def test_from_checkpoint_loads_saved_config_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            torch.save(
                {
                    "config": {"model": {"name": "custom/clip"}},
                    "model_state_dict": {"w": torch.ones(1)},
                },
                path,
            )

            tiny = _TinyModel()
            with mock.patch.object(
                CLIPDeepfakeDetector, "from_config", return_value=tiny
            ) as from_config:
                model = CLIPDeepfakeDetector.from_checkpoint(path)

        self.assertIsNotNone(model)
        self.assertTrue(torch.equal(model.w, torch.ones(1)))
        from_config.assert_called_once_with({"model": {"name": "custom/clip"}})


class BackendMetadataTests(unittest.TestCase):
    def test_fallback_metadata_does_not_claim_clip(self):
        version = image_model_version(False)
        self.assertEqual(version.modality, "image")
        self.assertEqual(version.status, "fallback")
        self.assertNotIn("CLIP", version.name)

    def test_active_metadata_claims_clip(self):
        version = image_model_version(True)
        self.assertEqual(version.status, "active")
        self.assertIn("CLIP", version.name)

    def test_image_construction_is_guarded_by_checkpoint_existence(self):
        source = (PROJECT_ROOT / "backend" / "inference.py").read_text()
        exists_at = source.index("settings.image_checkpoint.exists()")
        construct_at = source.index("CLIPDeepfakeDetector.from_checkpoint")
        self.assertLess(exists_at, construct_at)


if __name__ == "__main__":
    unittest.main()
