"""End-to-end integration tests for the image data pipeline.

These tests exercise the real manifest builder, the real splitter, and the real
dataset loader against a temporary GenImage-shaped directory tree containing
small generated images.

Run from the ``verisense`` directory with:

    .venv/bin/python -m unittest tests.test_image_pipeline_integration -v

The tests use only temporary directories and generated images. They do not
download datasets or model weights, and they do not train or evaluate a model.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.make_splits import SplitError, make_image_splits
from scripts.prepare_image_data import MANIFEST_COLUMNS, main as prepare_main
from src.image.dataset import DeepfakeImageDataset, ImageReadError


GENERATORS = ("midjourney", "sd_1_4", "adm", "biggan")
GENERATOR_FOLDERS = {
    "midjourney": "Midjourney",
    "sd_1_4": "Stable Diffusion V1.4",
    "sd_1_5": "Stable Diffusion V1.5",
    "adm": "ADM",
    "glide": "GLIDE",
    "wukong": "Wukong",
    "vqdm": "VQDM",
    "biggan": "BigGAN",
}
SOURCE_SPLITS = ("train", "val")
IMAGE_SPLITS = ("train", "val", "test")
IMAGE_SIZE = 48
FAKES_PER_GENERATOR_SPLIT = 2
REALS_PER_GENERATOR_SPLIT = 1
VAL_GENERATOR = "sd_1_4"
TEST_GENERATOR = "biggan"
TRAIN_GENERATORS = {"midjourney", "adm"}


class ImagePipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.workspace = self.root / "workspace"
        self.genimage_dir = (
            self.workspace / "data" / "raw" / "genimage"
        )
        self.processed_dir = self.workspace / "data" / "processed"
        self.splits_dir = self.workspace / "data" / "splits"
        self._color_index = 0

    def tearDown(self):
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # Fixture helpers
    # ------------------------------------------------------------------
    def _next_color(self):
        index = self._color_index
        self._color_index += 1
        return (
            index % 256,
            (index * 7 + 3) % 256,
            (index * 13 + 5) % 256,
        )

    def _write_image(self, relative_path: str) -> Path:
        path = self.genimage_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), self._next_color()).save(path)
        return path

    def _write_genimage_tree(self, generators=GENERATORS):
        for generator in generators:
            folder = GENERATOR_FOLDERS[generator]
            for source_split in SOURCE_SPLITS:
                for index in range(FAKES_PER_GENERATOR_SPLIT):
                    self._write_image(
                        f"{folder}/{source_split}/ai/fake_{index}.png"
                    )
                for index in range(REALS_PER_GENERATOR_SPLIT):
                    self._write_image(
                        f"{folder}/{source_split}/nature/real_{index}.png"
                    )

    def _expected_manifest_rows(self, generators) -> int:
        return (
            len(generators)
            * len(SOURCE_SPLITS)
            * (FAKES_PER_GENERATOR_SPLIT + REALS_PER_GENERATOR_SPLIT)
        )

    def build_manifest(self, generators=GENERATORS):
        """Run the real manifest builder CLI against a temporary GenImage tree."""
        self._write_genimage_tree(generators)
        expected_rows = self._expected_manifest_rows(generators)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            prepare_main(
                [
                    "--input-dir",
                    str(self.genimage_dir),
                    "--output-dir",
                    str(self.processed_dir / "images"),
                    "--data-root",
                    str(self.workspace),
                    "--dataset",
                    "genimage",
                    "--verify",
                ]
            )

        manifest_path = self.processed_dir / "images" / "genimage_manifest.csv"
        self.assertTrue(manifest_path.exists())
        self.assertIn(
            f"Manifest rows written: {expected_rows}",
            stdout.getvalue(),
        )
        return manifest_path

    def run_splitter(self, output_dir=None):
        """Run the real cross-generator splitter against the built manifest."""
        output_dir = output_dir or self.splits_dir
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = make_image_splits(
                self.processed_dir,
                output_dir,
                seed=42,
                protocol="cross_generator",
                val_generators=[VAL_GENERATOR],
                test_generators=[TEST_GENERATOR],
            )
        self.assertIsNotNone(metadata)
        return metadata

    def read_split(self, name: str, output_dir=None):
        output_dir = output_dir or self.splits_dir
        return pd.read_csv(output_dir / f"image_{name}.csv")

    def load_dataset(self, name: str, output_dir=None):
        output_dir = output_dir or self.splits_dir
        return DeepfakeImageDataset(
            output_dir / f"image_{name}.csv",
            transform=None,
            data_root=self.workspace,
        )

    # ------------------------------------------------------------------
    # Positive pipeline tests
    # ------------------------------------------------------------------
    def test_manifest_split_dataset_pipeline_end_to_end(self):
        manifest_path = self.build_manifest()
        manifest = pd.read_csv(manifest_path)
        expected_rows = self._expected_manifest_rows(GENERATORS)

        # --- Manifest builder output ---
        self.assertEqual(len(manifest), expected_rows)
        self.assertEqual(list(manifest.columns), MANIFEST_COLUMNS)
        self.assertEqual(set(manifest["dataset"]), {"genimage"})
        self.assertEqual(set(manifest["format"]), {"png"})
        self.assertEqual(set(manifest["grouping_basis"]), {"sha256_exact_content"})
        self.assertEqual(
            set(manifest["grouping_limitation"]),
            {"near_duplicates_and_shared_source_images_not_detected"},
        )
        self.assertTrue(manifest["group_id"].is_unique)
        self.assertEqual(
            set(manifest.loc[manifest["label"] == 0, "generator"]), {"real"}
        )
        self.assertEqual(
            set(manifest.loc[manifest["label"] == 0, "source"]), {"imagenet"}
        )
        self.assertEqual(
            set(manifest.loc[manifest["label"] == 1, "generator"]),
            set(GENERATORS),
        )

        for row in manifest.itertuples(index=False):
            # The builder stores portable paths relative to --data-root.
            self.assertFalse(Path(row.file_path).is_absolute())
            self.assertTrue(row.file_path.startswith("data/raw/genimage/"))
            self.assertEqual(row.label, 1 if "/ai/" in row.file_path else 0)
            self.assertEqual(
                row.source_split,
                "train" if "/train/" in row.file_path else "val",
            )

        # --- Real splitter output ---
        metadata = self.run_splitter()
        self.assertEqual(metadata["protocol"], "cross_generator")
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["source_split_policy"], "ignore")
        self.assertEqual(metadata["val_generators"], [VAL_GENERATOR])
        self.assertEqual(metadata["test_generators"], [TEST_GENERATOR])
        self.assertEqual(set(metadata["train_generators"]), TRAIN_GENERATORS)
        self.assertEqual(metadata["grouping_basis"], "sha256_exact_content")
        self.assertEqual(
            metadata["grouping_limitation"],
            "near_duplicates_and_shared_source_images_not_detected",
        )
        self.assertTrue(metadata["group_overlap_checked"])
        self.assertFalse(metadata["near_duplicate_detection"])
        self.assertFalse(metadata["source_level_grouping"])

        stored_metadata = json.loads(
            (self.splits_dir / "image_split_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stored_metadata["protocol"], "cross_generator")
        self.assertEqual(stored_metadata["seed"], 42)
        self.assertEqual(stored_metadata["val_generators"], [VAL_GENERATOR])
        self.assertEqual(stored_metadata["test_generators"], [TEST_GENERATOR])

        # --- Load every split through the real dataset ---
        manifest_by_path = manifest.set_index("file_path", drop=False)
        split_frames = {}
        group_sets = {}
        fake_generator_sets = {}

        for name in IMAGE_SPLITS:
            frame = self.read_split(name)
            split_frames[name] = frame
            group_sets[name] = set(frame["group_id"])
            fake_generator_sets[name] = set(
                frame.loc[frame["label"] == 1, "generator"]
            )

            self.assertEqual(list(frame.columns), MANIFEST_COLUMNS)
            self.assertEqual(len(frame), metadata["splits"][name]["rows"])
            self.assertEqual(
                set(frame["generator"]),
                set(metadata["splits"][name]["generators"]),
            )
            self.assertEqual(
                set(frame["source_split"]),
                set(metadata["splits"][name]["source_splits"]),
            )
            self.assertTrue(set(frame["label"]).issubset({0, 1}))
            self.assertTrue(
                (frame["label"] == 0).any(),
                f"{name} should contain real images",
            )

            dataset = self.load_dataset(name)
            self.assertEqual(len(dataset), len(frame))
            self.assertEqual(list(dataset.df.columns), MANIFEST_COLUMNS)

            for index, row in enumerate(frame.itertuples(index=False)):
                sample = dataset[index]

                self.assertEqual(
                    tuple(sample["image"].shape),
                    (3, IMAGE_SIZE, IMAGE_SIZE),
                )
                self.assertEqual(int(sample["label"]), int(row.label))
                self.assertEqual(sample["generator"], row.generator)

                resolved = Path(sample["file_path"])
                self.assertTrue(resolved.is_absolute())
                self.assertEqual(
                    resolved,
                    (self.workspace / row.file_path).resolve(),
                )
                self.assertTrue(resolved.is_file())

                # Every manifest field survives the splitter and the loader.
                for column in MANIFEST_COLUMNS:
                    self.assertEqual(
                        dataset.df.iloc[index][column],
                        manifest_by_path.loc[row.file_path, column],
                        f"{name} row {index}: column {column} changed",
                    )

        # No exact-content group crosses a split boundary.
        for index, first in enumerate(IMAGE_SPLITS):
            for second in IMAGE_SPLITS[index + 1 :]:
                self.assertFalse(
                    group_sets[first] & group_sets[second],
                    f"group overlap between {first} and {second}",
                )
        self.assertEqual(
            sum(len(groups) for groups in group_sets.values()),
            len(manifest),
        )

        # Cross-generator holdouts are strict.
        self.assertEqual(fake_generator_sets["val"], {VAL_GENERATOR})
        self.assertEqual(fake_generator_sets["test"], {TEST_GENERATOR})
        self.assertEqual(fake_generator_sets["train"], TRAIN_GENERATORS)
        for index, first in enumerate(IMAGE_SPLITS):
            for second in IMAGE_SPLITS[index + 1 :]:
                self.assertFalse(
                    fake_generator_sets[first] & fake_generator_sets[second],
                    f"fake-generator overlap between {first} and {second}",
                )

    def test_same_seed_produces_identical_split_outputs(self):
        self.build_manifest()
        first_dir = self.root / "splits_first"
        second_dir = self.root / "splits_second"

        first_metadata = self.run_splitter(output_dir=first_dir)
        second_metadata = self.run_splitter(output_dir=second_dir)

        self.assertEqual(first_metadata, second_metadata)
        for name in IMAGE_SPLITS:
            self.assertEqual(
                (first_dir / f"image_{name}.csv").read_bytes(),
                (second_dir / f"image_{name}.csv").read_bytes(),
                f"{name} split is not deterministic for the same seed",
            )
        self.assertEqual(
            (first_dir / "image_split_metadata.json").read_bytes(),
            (second_dir / "image_split_metadata.json").read_bytes(),
        )

    # ------------------------------------------------------------------
    # Negative tests
    # ------------------------------------------------------------------
    def test_dataset_raises_for_image_deleted_after_splitting(self):
        self.build_manifest()
        self.run_splitter()

        train_frame = self.read_split("train")
        relative_path = train_frame.iloc[0]["file_path"]
        victim = self.workspace / relative_path
        self.assertTrue(victim.is_file())
        victim.unlink()

        dataset = self.load_dataset("train")
        with self.assertRaises(ImageReadError) as context:
            _ = dataset[0]

        message = str(context.exception)
        self.assertIn("file not found", message)
        self.assertIn(relative_path, message)

    def test_splitter_rejects_invalid_label(self):
        manifest_path = self.build_manifest()
        manifest = pd.read_csv(manifest_path)
        manifest.loc[0, "label"] = 2
        manifest.to_csv(manifest_path, index=False)

        with self.assertRaisesRegex(
            SplitError, r"only 0 \(real\) or 1 \(fake\)"
        ):
            self.run_splitter()

    def test_splitter_rejects_malformed_manifest(self):
        manifest_path = self.build_manifest()
        manifest = pd.read_csv(manifest_path).drop(columns=["group_id"])
        manifest.to_csv(manifest_path, index=False)

        with self.assertRaisesRegex(
            SplitError, "missing required columns.*group_id"
        ):
            self.run_splitter()

    def test_cross_generator_rejects_too_few_generators(self):
        two_generators = ("midjourney", "sd_1_4")
        manifest_path = self.build_manifest(generators=two_generators)

        manifest = pd.read_csv(manifest_path)
        self.assertEqual(
            len(manifest),
            self._expected_manifest_rows(two_generators),
        )
        self.assertEqual(
            set(manifest.loc[manifest["label"] == 1, "generator"]),
            set(two_generators),
        )

        with self.assertRaisesRegex(SplitError, "at least 3 fake generators"):
            self.run_splitter()


if __name__ == "__main__":
    unittest.main()
