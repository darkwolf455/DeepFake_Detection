"""
VeriSense: Make Splits (Leakage-Aware Partitioning)

Image splitting supports two explicit protocols:

``cross_generator`` (default)
    Entire fake generators are held out for validation and test. A fake
    generator used for training never appears in validation or test. Real
    images are split separately, group-wise, across all three splits.

``in_distribution``
    The same generators may appear in every split, but ``group_id`` values are
    kept disjoint. By default the official GenImage ``source_split`` is
    respected: official ``train`` becomes train/val and official ``val``
    becomes test. Passing ``--source-split-policy ignore`` explicitly pools
    the rows instead.

Grouping limitation
-------------------
``group_id`` is treated as an opaque grouping key supplied by the manifest.
The current GenImage manifest uses SHA-256 exact-file-content identity only.
It does NOT detect near-duplicates or shared source images, so group-disjoint
splits do NOT guarantee source-level leakage prevention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


IMAGE_SPLITS = ("train", "val", "test")
IMAGE_PROTOCOLS = ("cross_generator", "in_distribution")
SOURCE_SPLIT_POLICIES = ("auto", "respect", "ignore")
REQUIRED_IMAGE_COLUMNS = (
    "dataset",
    "file_path",
    "label",
    "generator",
    "group_id",
    "source_split",
)
ROW_ORDER_COLUMN = "__verisense_row_order__"


class SplitError(ValueError):
    """Raised when a leakage-aware split cannot be constructed safely."""


def _value_counts(series: pd.Series) -> dict:
    counts = series.astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _describe_split(subset: pd.DataFrame) -> dict:
    return {
        "rows": int(len(subset)),
        "datasets": _value_counts(subset["dataset"]),
        "labels": _value_counts(subset["label"]),
        "generators": _value_counts(subset["generator"]),
        "source_splits": _value_counts(subset["source_split"]),
    }


def _manifest_metadata(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return "not_provided_by_manifest"
    values = sorted(
        {
            str(value).strip()
            for value in df[column].dropna().tolist()
            if str(value).strip()
        }
    )
    if not values:
        return "not_provided_by_manifest"
    return values[0] if len(values) == 1 else values


def validate_image_manifest(
    df: pd.DataFrame, allow_multiple_datasets: bool = False
) -> pd.DataFrame:
    """Validate and normalise an image manifest before splitting."""
    missing = [column for column in REQUIRED_IMAGE_COLUMNS if column not in df.columns]
    if missing:
        raise SplitError(
            f"Image manifest is missing required columns: {missing}. "
            f"Required columns: {list(REQUIRED_IMAGE_COLUMNS)}."
        )

    if df.empty:
        raise SplitError("Image manifest contains no rows; cannot build splits.")

    null_counts = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_IMAGE_COLUMNS
        if int(df[column].isna().sum()) > 0
    }
    if null_counts:
        raise SplitError(
            f"Image manifest contains missing values in required columns: {null_counts}."
        )

    work = df.copy()
    for column in ("dataset", "file_path", "generator", "group_id", "source_split"):
        work[column] = work[column].astype(str).str.strip()
        empty = work[column].eq("")
        if empty.any():
            raise SplitError(
                f"Image manifest contains {int(empty.sum())} empty value(s) in "
                f"required column '{column}'."
            )

    try:
        labels = pd.to_numeric(work["label"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SplitError("Column 'label' must contain numeric values 0 or 1.") from exc
    invalid_labels = sorted(set(labels[~labels.isin([0, 1])].tolist()))
    if invalid_labels:
        raise SplitError(
            "Column 'label' must contain only 0 (real) or 1 (fake); "
            f"found {invalid_labels}."
        )
    work["label"] = labels.astype(int)

    duplicated_rows = work.duplicated(keep=False)
    if duplicated_rows.any():
        examples = work.loc[duplicated_rows, "file_path"].head(5).tolist()
        raise SplitError(
            f"Image manifest contains {int(duplicated_rows.sum())} duplicate row(s). "
            f"Examples: {examples}. Remove exact duplicate rows before splitting."
        )

    duplicated_paths = work["file_path"].duplicated(keep=False)
    if duplicated_paths.any():
        examples = work.loc[duplicated_paths, "file_path"].head(5).tolist()
        raise SplitError(
            f"Image manifest contains {int(duplicated_paths.sum())} row(s) with "
            f"duplicate file_path values. Examples: {examples}."
        )

    datasets = sorted(work["dataset"].unique().tolist())
    if len(datasets) > 1 and not allow_multiple_datasets:
        raise SplitError(
            f"Image manifest contains multiple datasets {datasets}. This splitter "
            "does not merge unrelated datasets. Run it once per dataset, or pass "
            "--allow-multiple-datasets after reviewing the grouping."
        )

    labels_per_group = work.groupby("group_id")["label"].nunique()
    conflicting_groups = labels_per_group[labels_per_group > 1]
    if not conflicting_groups.empty:
        raise SplitError(
            f"{len(conflicting_groups)} group_id value(s) contain both real and fake "
            "labels. Exact-content groups must have a single label."
        )

    fake_real = (work["label"] == 1) & (work["generator"].str.lower() == "real")
    if fake_real.any():
        raise SplitError(
            "Rows with label=1 (fake) must not use generator='real'. "
            "Fix the manifest labelling before splitting."
        )

    return work


def normalize_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> dict:
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    if any(value <= 0 for value in ratios.values()):
        raise SplitError("All split ratios must be positive.")
    total = sum(ratios.values())
    if total <= 0:
        raise SplitError("Split ratios must sum to a positive value.")
    return {name: value / total for name, value in ratios.items()}


def _seeded_rng(seed: int, *parts) -> random.Random:
    payload = ":".join([str(seed), *[str(part) for part in parts]]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(value)


def _allocate_pool(
    pool: pd.DataFrame,
    split_names: Sequence[str],
    ratios: dict,
    seed: int,
    pool_name: str,
    warnings: list[str],
) -> dict:
    """Assign whole groups to splits, stratified by label.

    Every group stays in exactly one split. Each label is allocated
    independently so both classes are represented where enough groups exist.
    """
    if pool.empty:
        raise SplitError(f"The {pool_name} pool is empty; cannot construct splits.")

    selected_ratios = {name: ratios[name] for name in split_names}
    ratio_total = sum(selected_ratios.values())
    selected_ratios = {
        name: value / ratio_total for name, value in selected_ratios.items()
    }

    assigned_groups = {name: [] for name in split_names}

    for label in sorted(pool["label"].unique().tolist()):
        label_pool = pool[pool["label"] == label]
        group_sizes = label_pool.groupby("group_id", sort=True).size()
        groups = list(group_sizes.index)
        _seeded_rng(seed, pool_name, label).shuffle(groups)

        total_rows = int(group_sizes.sum())
        targets = {
            name: total_rows * selected_ratios[name] for name in split_names
        }
        current = {name: 0 for name in split_names}
        guaranteed: set = set()

        if len(groups) >= len(split_names):
            for name, group in zip(split_names, groups[: len(split_names)]):
                assigned_groups[name].append(group)
                current[name] += int(group_sizes.loc[group])
                guaranteed.add(group)

        for group in groups:
            if group in guaranteed:
                continue
            target_split = max(
                split_names,
                key=lambda name: (
                    targets[name] - current[name],
                    -list(split_names).index(name),
                ),
            )
            assigned_groups[target_split].append(group)
            current[target_split] += int(group_sizes.loc[group])

    subsets = {
        name: pool[pool["group_id"].isin(set(assigned_groups[name]))].copy()
        for name in split_names
    }

    for label in sorted(pool["label"].unique().tolist()):
        group_count = int(pool.loc[pool["label"] == label, "group_id"].nunique())
        missing_names = [
            name
            for name in split_names
            if not (subsets[name]["label"] == label).any()
        ]
        if group_count >= len(split_names):
            if missing_names:
                raise SplitError(
                    f"{pool_name}: label {label} has enough groups but is missing "
                    f"from split(s) {missing_names}. This indicates an allocation bug."
                )
        elif missing_names:
            warnings.append(
                f"{pool_name}: label {label} has only {group_count} group(s), so it "
                f"cannot appear in every split (missing: {missing_names})."
            )

    return subsets


def _parse_generator_list(value, label: str) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise SplitError(f"{label} generator list cannot be empty.")
    if len(set(items)) != len(items):
        raise SplitError(f"{label} generator list contains duplicates: {items}.")
    return items


def _choose_cross_generator_holdouts(
    available: Sequence[str],
    val_generators,
    test_generators,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    available = sorted(set(str(item) for item in available))
    if len(available) < 3:
        raise SplitError(
            "cross_generator requires at least 3 fake generators (training plus "
            f"distinct validation and test generators); found {len(available)}: "
            f"{available}. Use --protocol in_distribution if you do not have enough."
        )

    val = _parse_generator_list(val_generators, "Validation")
    test = _parse_generator_list(test_generators, "Test")

    for name, requested in (("validation", val), ("test", test)):
        if requested:
            absent = [generator for generator in requested if generator not in available]
            if absent:
                raise SplitError(
                    f"Requested {name} generator(s) not present in manifest: {absent}. "
                    f"Available fake generators: {available}."
                )

    if val and test:
        overlap = sorted(set(val) & set(test))
        if overlap:
            raise SplitError(
                "Validation and test generators must be distinct; "
                f"overlap: {overlap}."
            )

    shuffled = available[:]
    _seeded_rng(seed, "cross_generator_holdouts").shuffle(shuffled)

    if val is None and test is None:
        val = [shuffled[0]]
        test = [shuffled[1]]
    elif val is None:
        candidates = [generator for generator in shuffled if generator not in set(test)]
        if not candidates:
            raise SplitError(
                "Cannot choose a validation generator distinct from the test generators."
            )
        val = [candidates[0]]
    elif test is None:
        candidates = [generator for generator in shuffled if generator not in set(val)]
        if not candidates:
            raise SplitError(
                "Cannot choose a test generator distinct from the validation generators."
            )
        test = [candidates[0]]

    train = [
        generator
        for generator in available
        if generator not in set(val) and generator not in set(test)
    ]
    if not train:
        raise SplitError(
            "No fake generators remain for training after holding out validation "
            f"and test generators. Validation: {val}; test: {test}."
        )

    return train, val, test


def _resolve_source_split_policy(protocol: str, policy: str) -> str:
    if protocol not in IMAGE_PROTOCOLS:
        raise SplitError(
            f"Unknown image split protocol '{protocol}'. "
            f"Choose one of: {list(IMAGE_PROTOCOLS)}."
        )
    if policy not in SOURCE_SPLIT_POLICIES:
        raise SplitError(
            f"Unknown source-split policy '{policy}'. "
            f"Choose one of: {list(SOURCE_SPLIT_POLICIES)}."
        )
    if policy == "auto":
        return "ignore" if protocol == "cross_generator" else "respect"
    if protocol == "cross_generator" and policy == "respect":
        raise SplitError(
            "cross_generator cannot respect the official source_split: held-out "
            "generators must contribute all their official train and val rows to "
            "the held-out split. Use --source-split-policy auto or ignore."
        )
    return policy


def _fake_generators(subset: pd.DataFrame) -> set:
    fakes = subset[subset["label"] == 1]
    return set(fakes["generator"].tolist()) - {"real"}


def _validate_splits(
    splits: dict,
    expected_rows: int,
    protocol: str,
    warnings: list[str],
) -> None:
    for name in IMAGE_SPLITS:
        if name not in splits:
            raise SplitError(f"Missing required split '{name}'.")
        if splits[name].empty:
            raise SplitError(
                f"The '{name}' split is empty; adjust ratios, generators, or the "
                "source-split policy."
            )

    total_rows = sum(len(subset) for subset in splits.values())
    if total_rows != expected_rows:
        raise SplitError(
            f"Split row count mismatch: assigned {total_rows} rows but manifest has "
            f"{expected_rows}. Every manifest row must appear exactly once."
        )

    all_paths = [
        path for subset in splits.values() for path in subset["file_path"].tolist()
    ]
    if len(all_paths) != len(set(all_paths)):
        raise SplitError("Duplicate file_path values across splits; assignment is invalid.")

    all_groups = {
        name: set(splits[name]["group_id"].tolist()) for name in IMAGE_SPLITS
    }
    for index, first in enumerate(IMAGE_SPLITS):
        for second in IMAGE_SPLITS[index + 1 :]:
            overlap = all_groups[first] & all_groups[second]
            if overlap:
                raise SplitError(
                    f"Group leakage: {len(overlap)} group_id value(s) appear in both "
                    f"'{first}' and '{second}'. Examples: {sorted(overlap)[:3]}."
                )

    if protocol == "cross_generator":
        generators = {
            name: _fake_generators(splits[name]) for name in IMAGE_SPLITS
        }
        for name in IMAGE_SPLITS:
            if not generators[name]:
                raise SplitError(
                    f"cross_generator requires at least one fake generator in the "
                    f"'{name}' split; found none."
                )
        val_and_test = generators["val"] & generators["test"]
        if val_and_test:
            raise SplitError(
                "Validation and test fake generators must be distinct; "
                f"overlap: {sorted(val_and_test)}."
            )
        for held_out_name in ("val", "test"):
            overlap = generators["train"] & generators[held_out_name]
            if overlap:
                raise SplitError(
                    "Fake-generator leakage: training generators also appear in "
                    f"'{held_out_name}': {sorted(overlap)}."
                )

    for name in IMAGE_SPLITS:
        labels = sorted(splits[name]["label"].unique().tolist())
        if len(labels) == 1:
            warnings.append(
                f"Split '{name}' contains only label {labels[0]} "
                "(0=real, 1=fake); this reflects the available pool."
            )


def split_image_dataframe(
    df: pd.DataFrame,
    protocol: str = "cross_generator",
    seed: int = 42,
    val_generators=None,
    test_generators=None,
    source_split_policy: str = "auto",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    allow_multiple_datasets: bool = False,
) -> tuple[dict, dict]:
    """Split an image manifest into train/val/test DataFrames.

    Returns ``(splits, metadata)`` where ``splits`` maps split name to a
    DataFrame and ``metadata`` records the protocol, generators, counts, and
    grouping limitations.
    """
    resolved_policy = _resolve_source_split_policy(protocol, source_split_policy)
    work = validate_image_manifest(
        df, allow_multiple_datasets=allow_multiple_datasets
    )
    work[ROW_ORDER_COLUMN] = range(len(work))
    ratios = normalize_ratios(train_ratio, val_ratio, test_ratio)
    warnings: list[str] = []

    metadata = {
        "protocol": protocol,
        "seed": int(seed),
        "ratios": ratios,
        "source_split_policy": resolved_policy,
        "grouping_basis": _manifest_metadata(work, "grouping_basis"),
        "grouping_limitation": _manifest_metadata(work, "grouping_limitation"),
        "group_overlap_checked": True,
        "near_duplicate_detection": False,
        "source_level_grouping": False,
    }

    if protocol == "cross_generator":
        available = sorted(
            work.loc[work["label"] == 1, "generator"].unique().tolist()
        )
        train_generators, val_generators_chosen, test_generators_chosen = (
            _choose_cross_generator_holdouts(
                available, val_generators, test_generators, seed
            )
        )

        fakes = work[work["label"] == 1]
        reals = work[work["label"] == 0]
        if fakes.empty:
            raise SplitError("cross_generator requires at least one fake image.")

        splits = {
            "train": fakes[fakes["generator"].isin(train_generators)].copy(),
            "val": fakes[fakes["generator"].isin(val_generators_chosen)].copy(),
            "test": fakes[fakes["generator"].isin(test_generators_chosen)].copy(),
        }

        if reals.empty:
            warnings.append(
                "No real images are present; every split will contain only fakes."
            )
        else:
            real_splits = _allocate_pool(
                reals, IMAGE_SPLITS, ratios, seed, "real_images", warnings
            )
            for name in IMAGE_SPLITS:
                splits[name] = pd.concat(
                    [splits[name], real_splits[name]], ignore_index=True
                )

        metadata.update(
            {
                "train_generators": train_generators,
                "val_generators": val_generators_chosen,
                "test_generators": test_generators_chosen,
            }
        )
    else:
        if resolved_policy == "respect":
            source_values = set(work["source_split"].unique().tolist())
            if source_values != {"train", "val"}:
                raise SplitError(
                    "in_distribution with source-split policy 'respect' requires "
                    "source_split values 'train' and 'val'; found "
                    f"{sorted(source_values)}. Use --source-split-policy ignore to "
                    "pool rows explicitly."
                )

            spanning = work.groupby("group_id")["source_split"].nunique()
            spanning = spanning[spanning > 1]
            if not spanning.empty:
                raise SplitError(
                    f"{len(spanning)} group_id value(s) occur in both official "
                    "'train' and 'val' source splits; groups cannot stay disjoint "
                    "while respecting the official split. Resolve the grouping "
                    "conflict or use --source-split-policy ignore."
                )

            official_train = work[work["source_split"] == "train"].copy()
            official_val = work[work["source_split"] == "val"].copy()
            if official_train.empty:
                raise SplitError(
                    "Official 'train' source split is empty; cannot build train/val. "
                    "Use --source-split-policy ignore to pool rows explicitly."
                )
            if official_val.empty:
                raise SplitError(
                    "Official 'val' source split is empty; cannot build test. "
                    "Use --source-split-policy ignore to pool rows explicitly."
                )

            train_val = _allocate_pool(
                official_train,
                ("train", "val"),
                ratios,
                seed,
                "official_train",
                warnings,
            )
            splits = {
                "train": train_val["train"],
                "val": train_val["val"],
                "test": official_val,
            }
        else:
            splits = _allocate_pool(
                work, IMAGE_SPLITS, ratios, seed, "all_rows", warnings
            )

    for name in IMAGE_SPLITS:
        splits[name] = (
            splits[name]
            .sort_values(ROW_ORDER_COLUMN)
            .reset_index(drop=True)
        )

    _validate_splits(splits, expected_rows=len(work), protocol=protocol, warnings=warnings)

    metadata["splits"] = {
        name: _describe_split(splits[name]) for name in IMAGE_SPLITS
    }
    metadata["warnings"] = warnings
    return splits, metadata


def resolve_image_manifest(
    processed_dir: Path, manifest: Optional[str] = None
) -> Optional[Path]:
    if manifest is not None:
        path = Path(manifest)
        if not path.exists():
            raise SplitError(f"Image manifest not found: {path}")
        if not path.is_file():
            raise SplitError(f"Image manifest path is not a file: {path}")
        return path

    image_dir = Path(processed_dir) / "images"
    if not image_dir.exists():
        return None

    candidates = sorted(image_dir.glob("*_manifest.csv"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise SplitError(
            f"Multiple image manifests found in {image_dir}: "
            f"{[path.name for path in candidates]}. Pass --manifest to select one; "
            "this splitter does not merge unrelated manifests."
        )
    return candidates[0]


def make_image_splits(
    processed_dir: Path,
    output_dir: Path,
    seed: int = 42,
    manifest: Optional[str] = None,
    protocol: str = "cross_generator",
    val_generators=None,
    test_generators=None,
    source_split_policy: str = "auto",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    allow_multiple_datasets: bool = False,
) -> Optional[dict]:
    manifest_csv = resolve_image_manifest(processed_dir, manifest)
    if manifest_csv is None:
        print(
            "Skipping images: no *_manifest.csv found under "
            f"{Path(processed_dir) / 'images'}."
        )
        return None

    df = pd.read_csv(manifest_csv)
    splits, metadata = split_image_dataframe(
        df,
        protocol=protocol,
        seed=seed,
        val_generators=val_generators,
        test_generators=test_generators,
        source_split_policy=source_split_policy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        allow_multiple_datasets=allow_multiple_datasets,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in IMAGE_SPLITS:
        splits[name].drop(columns=[ROW_ORDER_COLUMN], errors="ignore").to_csv(
            output_dir / f"image_{name}.csv", index=False
        )

    metadata_path = output_dir / "image_split_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"\nImage manifest: {manifest_csv.resolve()}")
    print(f"Protocol: {metadata['protocol']}")
    print(f"Seed: {metadata['seed']}")
    print(f"Source-split policy: {metadata['source_split_policy']}")
    if metadata["protocol"] == "cross_generator":
        print(f"Training fake generators: {metadata['train_generators']}")
        print(f"Held-out validation generators: {metadata['val_generators']}")
        print(f"Held-out test generators: {metadata['test_generators']}")
        print(
            "Source-split handling: generator holdout overrides official source_split. "
            "Held-out generator rows (official train and val) go to the held-out split; "
            "remaining fake-generator rows (official train and val) go to training; "
            "real rows are pooled and split group-wise across train/val/test."
        )
    elif metadata["source_split_policy"] == "respect":
        print(
            "Source-split handling: official train -> train/val (group-aware); "
            "official val -> test."
        )
    else:
        print(
            "Source-split handling: official source_split intentionally ignored and "
            "rows pooled group-wise; source_split is preserved as metadata only."
        )

    for name in IMAGE_SPLITS:
        description = metadata["splits"][name]
        print(f"\n{name.upper()} split: {description['rows']} rows")
        print(f"  datasets: {description['datasets']}")
        print(f"  labels (0=real, 1=fake): {description['labels']}")
        print(f"  generators: {description['generators']}")
        print(f"  source_split: {description['source_splits']}")

    print("\nGROUPING BASIS / LIMITATION:")
    print(f"  grouping_basis: {metadata['grouping_basis']}")
    print(f"  grouping_limitation: {metadata['grouping_limitation']}")
    print("  group_id is treated as an opaque grouping key supplied by the manifest.")
    print(
        "  With the current GenImage manifest it is SHA-256 exact-file-content "
        "identity only."
    )
    print("  It does NOT detect near-duplicates or shared source images.")
    print(
        "  Group-disjoint splits therefore do NOT guarantee source-level leakage "
        "prevention."
    )

    if metadata["warnings"]:
        print("\nWARNINGS:")
        for warning in metadata["warnings"]:
            print(f"  - {warning}")

    print(f"\nImage splits written to: {output_dir.resolve()}")
    print(f"Split metadata: {metadata_path.resolve()}")
    return metadata


def make_video_splits(processed_dir: Path, output_dir: Path, seed: int = 42):
    manifest_csv = processed_dir / "video_cache" / "video_manifest.csv"
    if not manifest_csv.exists():
        print(f"Skipping video: {manifest_csv} does not exist.")
        return

    df = pd.read_csv(manifest_csv)
    # Extract identity or parent video id (e.g. 001 from 001_002 or 001)
    df["group_id"] = df["video_id"].apply(lambda x: str(x).split("_")[0])

    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(df, groups=df["group_id"]))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["group_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    train_df.to_csv(output_dir / "video_train.csv", index=False)
    val_df.to_csv(output_dir / "video_val.csv", index=False)
    test_df.to_csv(output_dir / "video_test.csv", index=False)
    print(f"Video splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def make_audio_splits(processed_dir: Path, output_dir: Path, seed: int = 42):
    manifest_csv = processed_dir / "audio" / "audio_manifest.csv"
    if not manifest_csv.exists():
        print(f"Skipping audio: {manifest_csv} does not exist.")
        return

    df = pd.read_csv(manifest_csv)
    # Group by speaker_id to ensure zero speaker overlap
    groups = df["speaker_id"].fillna("unknown")

    gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(df, groups=groups))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["speaker_id"]))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    train_df.to_csv(output_dir / "audio_train.csv", index=False)
    val_df.to_csv(output_dir / "audio_val.csv", index=False)
    test_df.to_csv(output_dir / "audio_test.csv", index=False)
    print(f"Audio splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate leakage-aware train/val/test splits"
    )
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Explicit image manifest CSV. If omitted, exactly one *_manifest.csv "
            "under <processed-dir>/images is used."
        ),
    )
    parser.add_argument(
        "--protocol",
        choices=list(IMAGE_PROTOCOLS),
        default="cross_generator",
        help="Image split protocol (default: cross_generator).",
    )
    parser.add_argument(
        "--val-generators",
        type=str,
        default=None,
        help="Comma-separated fake generators held out for validation.",
    )
    parser.add_argument(
        "--test-generators",
        type=str,
        default=None,
        help="Comma-separated fake generators held out for test.",
    )
    parser.add_argument(
        "--source-split-policy",
        choices=list(SOURCE_SPLIT_POLICIES),
        default="auto",
        help=(
            "How to handle official source_split values. auto uses 'ignore' for "
            "cross_generator and 'respect' for in_distribution."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--allow-multiple-datasets",
        action="store_true",
        help="Allow a manifest containing more than one dataset value.",
    )
    args = parser.parse_args(argv)

    proc_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating splits in {out_dir} with seed={args.seed}...")
    try:
        make_image_splits(
            proc_dir,
            out_dir,
            seed=args.seed,
            manifest=args.manifest,
            protocol=args.protocol,
            val_generators=args.val_generators,
            test_generators=args.test_generators,
            source_split_policy=args.source_split_policy,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            allow_multiple_datasets=args.allow_multiple_datasets,
        )
    except SplitError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    make_video_splits(proc_dir, out_dir, args.seed)
    make_audio_splits(proc_dir, out_dir, args.seed)
    print(
        "All requested splits generated. Review the reported grouping limitations "
        "before training."
    )


if __name__ == "__main__":
    main()
