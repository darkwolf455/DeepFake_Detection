"""Focused tests for image dataset path resolution.

Run from the ``verisense`` directory with:

    .venv/bin/python -m unittest tests.test_image_dataset_paths -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.common.config import load_config
from src.image.dataset import DeepfakeImageDataset, ImageReadError


class ImageDatasetPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_image(self, relative_path: str, size=(32, 32), color=(10, 20, 30)):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    def write_split(self, file_paths, labels=None, filename="split.csv"):
        labels = labels or [1] * len(file_paths)
        rows = [
            {"file_path": file_path, "label": label, "generator": "test"}
            for file_path, label in zip(file_paths, labels)
        ]
        csv_path = self.root / filename
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_relative_path_resolves_against_data_root_with_spaces(self):
        data_root = self.root / "root with spaces"
        image = self.make_image("root with spaces/images/sample one.png")
        csv_path = self.write_split(["images/sample one.png"])

        dataset = DeepfakeImageDataset(
            csv_path, transform=None, data_root=data_root
        )
        sample = dataset[0]

        self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
        self.assertEqual(Path(sample["file_path"]), image.resolve())
        self.assertTrue(Path(sample["file_path"]).is_absolute())

    def test_absolute_path_is_preserved_without_data_root(self):
        image = self.make_image("absolute/real.png")
        csv_path = self.write_split([str(image)])

        dataset = DeepfakeImageDataset(csv_path, transform=None)
        sample = dataset[0]

        self.assertEqual(Path(sample["file_path"]), image.resolve())
        self.assertTrue(Path(sample["file_path"]).is_absolute())

    def test_absolute_path_does_not_require_existing_data_root(self):
        image = self.make_image("absolute/outside.png")
        csv_path = self.write_split([str(image)])

        dataset = DeepfakeImageDataset(
            csv_path,
            transform=None,
            data_root=self.root / "root does not need to exist",
        )
        sample = dataset[0]

        self.assertEqual(Path(sample["file_path"]), image.resolve())

    def test_relative_path_without_data_root_is_rejected(self):
        csv_path = self.write_split(["images/sample.png"])

        with self.assertRaisesRegex(ValueError, "data_root was not provided"):
            DeepfakeImageDataset(csv_path, transform=None)

    def test_missing_data_root_is_rejected(self):
        csv_path = self.write_split(["images/sample.png"])

        with self.assertRaisesRegex(FileNotFoundError, "data_root does not exist"):
            DeepfakeImageDataset(
                csv_path,
                transform=None,
                data_root=self.root / "missing root",
            )

    def test_data_root_that_is_not_a_directory_is_rejected(self):
        not_a_root = self.root / "not a directory"
        not_a_root.write_text("x", encoding="utf-8")
        csv_path = self.write_split(["images/sample.png"])

        with self.assertRaisesRegex(NotADirectoryError, "not a directory"):
            DeepfakeImageDataset(csv_path, transform=None, data_root=not_a_root)

    def test_missing_image_raises_clear_error(self):
        csv_path = self.write_split(["images/missing.png"])
        dataset = DeepfakeImageDataset(
            csv_path, transform=None, data_root=self.root
        )

        with self.assertRaises(ImageReadError) as context:
            _ = dataset[0]

        message = str(context.exception)
        self.assertIn("file not found", message)
        self.assertIn("images/missing.png", message)
        self.assertIn(str(self.root.resolve()), message)

    def test_corrupt_or_unreadable_image_raises_clear_error(self):
        bad_path = self.root / "images" / "corrupt.jpg"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_bytes(b"this is not an image")

        csv_path = self.write_split(["images/corrupt.jpg"])
        dataset = DeepfakeImageDataset(
            csv_path, transform=None, data_root=self.root
        )

        with self.assertRaisesRegex(ImageReadError, "unreadable or corrupt"):
            _ = dataset[0]

    def test_resolved_paths_survive_working_directory_changes(self):
        data_root = self.root / "data root"
        image = self.make_image("data root/images/pic one.png")
        csv_path = self.write_split(["images/pic one.png"])
        other_dir = self.root / "elsewhere"
        other_dir.mkdir()

        previous_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            dataset = DeepfakeImageDataset(
                csv_path.resolve(),
                transform=None,
                data_root="data root",
            )
            os.chdir(other_dir)
            sample = dataset[0]
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(Path(sample["file_path"]), image.resolve())
        self.assertTrue(Path(sample["file_path"]).is_absolute())

    def test_image_config_defines_an_explicit_data_root(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "image.yaml"
        config = load_config(config_path)

        self.assertIn("data_root", config.data)
        self.assertTrue(str(config.data.data_root).strip())


if __name__ == "__main__":
    unittest.main()
