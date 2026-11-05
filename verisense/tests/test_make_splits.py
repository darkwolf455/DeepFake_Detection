"""Focused tests for leakage-aware image splitting.

Run from the ``verisense`` directory with:

    .venv/bin/python -m unittest tests.test_make_splits -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.make_splits import (
    SplitError,
    make_image_splits,
    split_image_dataframe,
)

GENERATORS = ("midjourney", "sd_1_4", "sd_1_5", "biggan")


def image_row(
    file_path,
    label,
    generator,
    source_split,
    group_id=None,
    dataset="genimage",
):
    return {
        "dataset": dataset,
        "file_path": file_path,
        "label": label,
        "generator": generator,
        "source": "imagenet" if label == 0 else f"{dataset}/{generator}",
        "group_id": group_id or f"sha256:{file_path}",
        "source_split": source_split,
        "grouping_basis": "sha256_exact_content",
        "grouping_limitation": (
            "near_duplicates_and_shared_source_images_not_detected"
        ),
    }


def build_manifest(
    fake_generators=GENERATORS,
    source_splits=("train", "val"),
    fakes_per_generator_split=2,
    reals_per_split=3,
):
    rows = []
    for generator in fake_generators:
        for source_split in source_splits:
            for index in range(fakes_per_generator_split):
                rows.append(
                    image_row(
                        f"{generator}/{source_split}/ai/{index}.jpg",
                        1,
                        generator,
                        source_split,
                    )
                )
    for source_split in source_splits:
        for index in range(reals_per_split):
            rows.append(
                image_row(
                    f"real/{source_split}/nature/{index}.jpg",
                    0,
                    "real",
                    source_split,
                    group_id=f"sha256:real-{source_split}-{index}",
                )
            )
    return pd.DataFrame(rows)


class ImageSplitTests(unittest.TestCase):
    def assert_no_group_overlap(self, splits):
        seen = {}
        for name, subset in splits.items():
            for group_id in subset["group_id"]:
                if group_id in seen:
                    self.assertEqual(
                        seen[group_id],
                        name,
                        f"group {group_id} appears in both {seen[group_id]} and {name}",
                    )
                else:
                    seen[group_id] = name

    def test_cross_generator_holds_out_whole_generators(self):
        splits, metadata = split_image_dataframe(
            build_manifest(),
            protocol="cross_generator",
            seed=42,
            val_generators=["sd_1_4"],
            test_generators=["biggan"],
        )

        fake = {name: subset[subset["label"] == 1] for name, subset in splits.items()}
        self.assertEqual(set(fake["val"]["generator"]), {"sd_1_4"})
        self.assertEqual(set(fake["test"]["generator"]), {"biggan"})
        self.assertNotIn("sd_1_4", set(fake["train"]["generator"]))
        self.assertNotIn("biggan", set(fake["train"]["generator"]))

        train_generators = set(fake["train"]["generator"])
        self.assertFalse(train_generators & set(fake["val"]["generator"]))
        self.assertFalse(train_generators & set(fake["test"]["generator"]))
        self.assertFalse(
            set(fake["val"]["generator"]) & set(fake["test"]["generator"])
        )

        self.assert_no_group_overlap(splits)
        self.assertEqual(metadata["protocol"], "cross_generator")
        self.assertEqual(metadata["val_generators"], ["sd_1_4"])
        self.assertEqual(metadata["test_generators"], ["biggan"])

    def test_cross_generator_heldout_keeps_train_and_val_source_splits(self):
        splits, _ = split_image_dataframe(
            build_manifest(),
            protocol="cross_generator",
            seed=42,
            val_generators=["sd_1_4"],
            test_generators=["biggan"],
        )

        for name in ("val", "test"):
            fake_rows = splits[name][splits[name]["label"] == 1]
            self.assertEqual(set(fake_rows["source_split"]), {"train", "val"})

    def test_cross_generator_real_groups_stay_together(self):
        df = build_manifest()
        shared = pd.DataFrame(
            [
                image_row(
                    "real/train/nature/shared-a.jpg",
                    0,
                    "real",
                    "train",
                    group_id="sha256:shared-real",
                ),
                image_row(
                    "real/train/nature/shared-b.jpg",
                    0,
                    "real",
                    "train",
                    group_id="sha256:shared-real",
                ),
            ]
        )
        df = pd.concat([df, shared], ignore_index=True)

        splits, _ = split_image_dataframe(
            df,
            protocol="cross_generator",
            seed=42,
            val_generators=["sd_1_4"],
            test_generators=["biggan"],
        )

        locations = [
            name
            for name, subset in splits.items()
            if (subset["group_id"] == "sha256:shared-real").any()
        ]
        self.assertEqual(len(locations), 1)
        self.assertEqual(
            int((splits[locations[0]]["group_id"] == "sha256:shared-real").sum()),
            2,
        )
        self.assert_no_group_overlap(splits)

    def test_cross_generator_real_images_in_each_split_when_possible(self):
        splits, metadata = split_image_dataframe(
            build_manifest(),
            protocol="cross_generator",
            seed=42,
            val_generators=["sd_1_4"],
            test_generators=["biggan"],
        )

        for name in ("train", "val", "test"):
            self.assertTrue(
                (splits[name]["label"] == 0).any(),
                f"{name} has no real images",
            )
            self.assertEqual(metadata["splits"][name]["labels"].get("0", 0) > 0, True)

    def test_cross_generator_requires_three_generators(self):
        with self.assertRaisesRegex(SplitError, "at least 3 fake generators"):
            split_image_dataframe(
                build_manifest(fake_generators=("midjourney", "sd_1_4")),
                protocol="cross_generator",
            )

    def test_absent_requested_generator_fails(self):
        with self.assertRaisesRegex(SplitError, "not present in manifest"):
            split_image_dataframe(
                build_manifest(),
                protocol="cross_generator",
                val_generators=["does_not_exist"],
            )

    def test_validation_and_test_generators_must_differ(self):
        with self.assertRaisesRegex(SplitError, "must be distinct"):
            split_image_dataframe(
                build_manifest(),
                protocol="cross_generator",
                val_generators=["sd_1_4"],
                test_generators=["sd_1_4"],
            )

    def test_deterministic_for_same_seed(self):
        kwargs = dict(
            protocol="cross_generator",
            seed=17,
            val_generators=["sd_1_4"],
            test_generators=["biggan"],
        )
        first_splits, first_metadata = split_image_dataframe(build_manifest(), **kwargs)
        second_splits, second_metadata = split_image_dataframe(build_manifest(), **kwargs)

        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(first_splits[name], second_splits[name])
        self.assertEqual(first_metadata, second_metadata)

    def test_in_distribution_respects_official_source_split(self):
        splits, metadata = split_image_dataframe(
            build_manifest(),
            protocol="in_distribution",
            seed=42,
            source_split_policy="auto",
        )

        self.assertEqual(metadata["source_split_policy"], "respect")
        self.assertEqual(set(splits["train"]["source_split"]), {"train"})
        self.assertEqual(set(splits["val"]["source_split"]), {"train"})
        self.assertEqual(set(splits["test"]["source_split"]), {"val"})
        self.assert_no_group_overlap(splits)

    def test_in_distribution_ignore_is_explicit_and_group_aware(self):
        splits, metadata = split_image_dataframe(
            build_manifest(),
            protocol="in_distribution",
            seed=42,
            source_split_policy="ignore",
        )

        self.assertEqual(metadata["source_split_policy"], "ignore")
        self.assert_no_group_overlap(splits)
        for name in ("train", "val", "test"):
            self.assertTrue(
                set(splits[name]["source_split"]).issubset({"train", "val"})
            )

    def test_in_distribution_respect_fails_without_official_val_pool(self):
        with self.assertRaisesRegex(SplitError, "source_split values|source split"):
            split_image_dataframe(
                build_manifest(source_splits=("train",)),
                protocol="in_distribution",
                source_split_policy="respect",
            )

    def test_group_spanning_official_train_and_val_is_rejected(self):
        df = build_manifest()
        train_row = df[df["source_split"] == "train"].iloc[0]
        val_index = df[df["source_split"] == "val"].index[0]
        df.loc[val_index, "group_id"] = train_row["group_id"]

        with self.assertRaisesRegex(SplitError, "both official"):
            split_image_dataframe(
                df,
                protocol="in_distribution",
                source_split_policy="respect",
            )

    def test_missing_group_id_is_rejected(self):
        df = build_manifest()
        df.loc[0, "group_id"] = None

        with self.assertRaisesRegex(SplitError, "missing values.*group_id"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_empty_group_id_is_rejected(self):
        df = build_manifest()
        df.loc[0, "group_id"] = "   "

        with self.assertRaisesRegex(SplitError, "empty value.*group_id"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_group_with_conflicting_labels_is_rejected(self):
        df = build_manifest()
        real_group = df[df["label"] == 0].iloc[0]["group_id"]
        # File paths stay distinct, so this is a grouping conflict rather than
        # an exact duplicate-row conflict.
        df.loc[df[df["label"] == 1].index[0], "group_id"] = real_group

        with self.assertRaisesRegex(SplitError, "both real and fake"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_missing_required_column_is_rejected(self):
        df = build_manifest().drop(columns=["group_id"])

        with self.assertRaisesRegex(SplitError, "missing required columns.*group_id"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_invalid_label_is_rejected(self):
        df = build_manifest()
        df.loc[0, "label"] = 2

        with self.assertRaisesRegex(SplitError, "only 0"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_duplicate_rows_are_rejected(self):
        df = build_manifest()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(SplitError, "duplicate row"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_missing_source_split_is_rejected(self):
        df = build_manifest()
        df.loc[0, "source_split"] = None

        with self.assertRaisesRegex(SplitError, "missing values.*source_split"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_multiple_datasets_are_rejected_by_default(self):
        df = build_manifest()
        df.loc[df.index[0], "dataset"] = "other_dataset"

        with self.assertRaisesRegex(SplitError, "multiple datasets"):
            split_image_dataframe(df, protocol="cross_generator")

    def test_make_image_splits_writes_files_metadata_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "processed" / "images"
            image_dir.mkdir(parents=True)
            build_manifest().to_csv(image_dir / "genimage_manifest.csv", index=False)

            output_dir = root / "splits"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                metadata = make_image_splits(
                    root / "processed",
                    output_dir,
                    seed=42,
                    val_generators=["sd_1_4"],
                    test_generators=["biggan"],
                )

            for name in ("train", "val", "test"):
                self.assertTrue((output_dir / f"image_{name}.csv").exists())
            self.assertTrue((output_dir / "image_split_metadata.json").exists())

            written = pd.read_csv(output_dir / "image_train.csv")
            self.assertIn("source_split", written.columns)
            self.assertNotIn("__verisense_row_order__", written.columns)
            self.assertEqual(metadata["protocol"], "cross_generator")

            stored_metadata = json.loads(
                (output_dir / "image_split_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored_metadata["protocol"], "cross_generator")
            self.assertTrue(stored_metadata["group_overlap_checked"])
            self.assertFalse(stored_metadata["near_duplicate_detection"])

            printed = stdout.getvalue()
            for expected in (
                "GROUPING BASIS / LIMITATION",
                "grouping_basis",
                "source_split",
                "generators",
                "SHA-256 exact-file-content",
                "does NOT detect near-duplicates",
            ):
                self.assertIn(expected, printed)

    def test_multiple_manifests_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "processed" / "images"
            image_dir.mkdir(parents=True)
            build_manifest().to_csv(image_dir / "one_manifest.csv", index=False)
            build_manifest().to_csv(image_dir / "two_manifest.csv", index=False)

            with self.assertRaisesRegex(SplitError, "Multiple image manifests"):
                make_image_splits(root / "processed", root / "splits")


if __name__ == "__main__":
    unittest.main()
