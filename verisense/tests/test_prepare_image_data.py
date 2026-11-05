"""Focused tests for the GenImage manifest builder.

Run from the ``verisense`` directory with:

    .venv/bin/python -m unittest tests.test_prepare_image_data -v
"""

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.prepare_image_data import (
    EmptyScanError,
    GenImageLayoutError,
    ManifestError,
    REQUIRED_COLUMNS,
    ScanStats,
    canonical_generator,
    main,
    scan_image_directory,
)


class GenImageManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, relative_path: str, content: bytes | None = None) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if content is not None else relative_path.encode("utf-8"))
        return path

    def test_official_layout_labels_generators_and_source_split(self):
        self.write("GenImage/Midjourney/train/ai/fake_train.jpg", b"fake-train")
        self.write("GenImage/Midjourney/train/nature/real_train.jpg", b"real-train")
        self.write("GenImage/Midjourney/val/ai/fake_val.jpg", b"fake-val")
        self.write("GenImage/Midjourney/val/nature/real_val.jpg", b"real-val")

        records = scan_image_directory(self.root, data_root=self.root)
        self.assertEqual(len(records), 4)

        by_name = {Path(record["file_path"]).name: record for record in records}

        fake_train = by_name["fake_train.jpg"]
        self.assertEqual(fake_train["dataset"], "genimage")
        self.assertEqual(fake_train["label"], 1)
        self.assertEqual(fake_train["generator"], "midjourney")
        self.assertEqual(fake_train["source"], "genimage/Midjourney")
        self.assertEqual(fake_train["source_split"], "train")
        self.assertTrue(fake_train["group_id"].startswith("sha256:"))
        self.assertEqual(fake_train["grouping_basis"], "sha256_exact_content")
        self.assertIn("not_detected", fake_train["grouping_limitation"])

        real_train = by_name["real_train.jpg"]
        self.assertEqual(real_train["label"], 0)
        self.assertEqual(real_train["generator"], "real")
        self.assertEqual(real_train["source"], "imagenet")
        self.assertEqual(real_train["source_split"], "train")

        self.assertEqual(by_name["fake_val.jpg"]["source_split"], "val")
        self.assertEqual(by_name["real_val.jpg"]["source_split"], "val")

        for record in records:
            self.assertFalse(Path(record["file_path"]).is_absolute())
            self.assertTrue(set(REQUIRED_COLUMNS).issubset(record))

    def test_archive_layout_extracts_canonical_generator(self):
        self.write(
            "GenImage/imagenet_ai_0419_sdv4/train/ai/fake.jpg",
            b"archive-fake",
        )
        records = scan_image_directory(self.root, data_root=self.root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["generator"], "sd_1_4")
        self.assertEqual(records[0]["source"], "genimage/imagenet_ai_0419_sdv4")
        self.assertEqual(records[0]["label"], 1)
        self.assertEqual(records[0]["source_split"], "train")

    def test_canonical_generator_aliases(self):
        cases = {
            "Midjourney": "midjourney",
            "Stable Diffusion V1.4": "sd_1_4",
            "Stable Diffusion V1.5": "sd_1_5",
            "stable_diffusion_v_1_4": "sd_1_4",
            "stable_diffusion_v_1_5": "sd_1_5",
            "imagenet_ai_0419_sdv4": "sd_1_4",
            "imagenet_ai_0419_sdv5": "sd_1_5",
            "ADM": "adm",
            "imagenet_ai_0508_adm": "adm",
            "GLIDE": "glide",
            "Wukong": "wukong",
            "VQDM": "vqdm",
            "VQ-Diffusion": "vqdm",
            "BigGAN": "biggan",
        }

        for folder_name, expected in cases.items():
            with self.subTest(folder_name=folder_name):
                self.assertEqual(canonical_generator(folder_name), expected)

    def test_unknown_canonical_generators_are_rejected(self):
        for folder_name in ["DALL-E", "SD", "Stable Diffusion", "unknown-generator"]:
            with self.subTest(folder_name=folder_name):
                with self.assertRaises(GenImageLayoutError):
                    canonical_generator(folder_name)

    def test_labels_come_from_ai_nature_not_misleading_parents(self):
        # The input directory itself is named "ai"; that must not turn a
        # nature image into a fake image.
        input_dir = self.root / "ai"
        self.write("ai/GenImage/Midjourney/train/nature/real.jpg", b"real")
        self.write("ai/GenImage/Midjourney/val/ai/fake.jpg", b"fake")
        self.write("ai/fake/GenImage/Midjourney/train/nature/real2.jpg", b"real2")

        records = scan_image_directory(input_dir, data_root=self.root)
        by_name = {Path(record["file_path"]).name: record for record in records}

        self.assertEqual(by_name["real.jpg"]["label"], 0)
        self.assertEqual(by_name["fake.jpg"]["label"], 1)
        self.assertEqual(by_name["real2.jpg"]["label"], 0)

        self.assertEqual(by_name["real.jpg"]["generator"], "real")
        self.assertEqual(by_name["fake.jpg"]["generator"], "midjourney")
        self.assertEqual(by_name["real2.jpg"]["generator"], "real")

    def test_input_directory_can_be_a_generator_folder(self):
        self.write("Midjourney/train/ai/fake.jpg", b"fake")
        self.write("Midjourney/val/nature/real.jpg", b"real")

        records = scan_image_directory(self.root / "Midjourney", data_root=self.root)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record["generator"] for record in records},
            {"midjourney", "real"},
        )
        self.assertEqual({record["source_split"] for record in records}, {"train", "val"})
        self.assertEqual(
            {record["source"] for record in records},
            {"genimage/Midjourney", "imagenet"},
        )

    def test_input_directory_can_be_a_split_folder(self):
        self.write("Midjourney/train/ai/fake.jpg", b"fake")
        self.write("Midjourney/train/nature/real.jpg", b"real")

        records = scan_image_directory(self.root / "Midjourney" / "train", data_root=self.root)
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {record["generator"] for record in records},
            {"midjourney", "real"},
        )
        self.assertEqual({record["source_split"] for record in records}, {"train"})

    def test_flat_readme_style_layout_is_rejected(self):
        self.write("GenImage/midjourney/x.jpg", b"x")

        with self.assertRaisesRegex(GenImageLayoutError, "Unrecognized GenImage layout"):
            scan_image_directory(self.root, data_root=self.root)

    def test_unknown_split_is_rejected(self):
        self.write("GenImage/Midjourney/test/ai/x.jpg", b"x")

        with self.assertRaisesRegex(GenImageLayoutError, "Expected 'train' or 'val'"):
            scan_image_directory(self.root, data_root=self.root)

    def test_unknown_label_folder_is_rejected(self):
        self.write("GenImage/Midjourney/train/not-a-label/x.jpg", b"x")

        with self.assertRaisesRegex(GenImageLayoutError, "Unrecognized GenImage layout"):
            scan_image_directory(self.root, data_root=self.root)

    def test_unknown_generator_is_rejected(self):
        self.write("GenImage/DALL-E/train/ai/x.jpg", b"x")

        with self.assertRaisesRegex(
            GenImageLayoutError, "Unrecognized GenImage generator"
        ):
            scan_image_directory(self.root, data_root=self.root)

    def test_merged_imagenet_ai_layout_is_rejected(self):
        self.write("GenImage/imagenet_ai/train/ai/x.jpg", b"x")

        with self.assertRaisesRegex(GenImageLayoutError, "generator identity"):
            scan_image_directory(self.root, data_root=self.root)

    def test_empty_scan_raises_clear_error(self):
        with self.assertRaisesRegex(EmptyScanError, "No image files found"):
            scan_image_directory(self.root, data_root=self.root)

    def test_exact_duplicate_contents_are_deduplicated(self):
        self.write("GenImage/Midjourney/train/ai/a.jpg", b"same-content")
        self.write("GenImage/Midjourney/train/ai/b.jpg", b"same-content")

        stats = ScanStats()
        records = scan_image_directory(self.root, data_root=self.root, stats=stats)

        self.assertEqual(len(records), 1)
        self.assertEqual(Path(records[0]["file_path"]).name, "a.jpg")
        self.assertEqual(stats.duplicate_contents, 1)
        expected_group = "sha256:" + hashlib.sha256(b"same-content").hexdigest()
        self.assertEqual(records[0]["group_id"], expected_group)

    def test_duplicate_paths_via_symlink_are_deduplicated(self):
        original = self.write(
            "GenImage/Midjourney/train/ai/original.jpg",
            b"unique-content",
        )
        link = self.root / "GenImage/Midjourney/train/ai/link.jpg"

        try:
            link.symlink_to(original)
        except OSError:
            self.skipTest("symlinks are not supported on this platform")

        stats = ScanStats()
        records = scan_image_directory(self.root, data_root=self.root, stats=stats)

        self.assertEqual(len(records), 1)
        self.assertEqual(Path(records[0]["file_path"]).name, "original.jpg")
        self.assertEqual(stats.duplicate_paths, 1)

    def test_duplicate_content_with_conflicting_labels_is_rejected(self):
        self.write("GenImage/Midjourney/train/ai/fake.jpg", b"same-content")
        self.write("GenImage/Midjourney/train/nature/real.jpg", b"same-content")

        with self.assertRaisesRegex(ManifestError, "conflicting labels"):
            scan_image_directory(self.root, data_root=self.root)

    def test_file_paths_are_relative_to_data_root(self):
        self.write("data/raw/genimage/Midjourney/train/ai/x.jpg", b"x")

        records = scan_image_directory(
            self.root / "data" / "raw" / "genimage",
            data_root=self.root / "data" / "raw",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["file_path"],
            "genimage/Midjourney/train/ai/x.jpg",
        )
        self.assertFalse(Path(records[0]["file_path"]).is_absolute())

    def test_main_writes_manifest_and_prints_counts(self):
        self.write("GenImage/Midjourney/train/ai/fake.jpg", b"fake")
        self.write("GenImage/Midjourney/train/nature/real.jpg", b"real")
        output_dir = self.root / "processed"

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(
                [
                    "--input-dir",
                    str(self.root),
                    "--output-dir",
                    str(output_dir),
                    "--data-root",
                    str(self.root),
                    "--dataset",
                    "genimage",
                ]
            )

        manifest_path = output_dir / "genimage_manifest.csv"
        self.assertTrue(manifest_path.exists())

        df = pd.read_csv(manifest_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns)[: len(REQUIRED_COLUMNS)], REQUIRED_COLUMNS)

        printed = stdout.getvalue()
        for expected_text in [
            "Dataset counts",
            "Label counts",
            "Generator counts",
            "Official source_split counts",
            "Source counts",
            "GROUPING LIMITATION",
            "near-duplicates",
        ]:
            self.assertIn(expected_text, printed)


if __name__ == "__main__":
    unittest.main()
