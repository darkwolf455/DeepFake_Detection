"""
VeriSense: Prepare Image Data (GenImage manifest builder)

Walks an official GenImage directory tree and writes a portable CSV manifest.

Supported GenImage layouts
--------------------------
1. Root containing one or more generator folders:
       <input>/<Generator>/<train|val>/<ai|nature>/<image>
2. A single generator folder:
       <input>/<train|val>/<ai|nature>/<image>
3. An extracted per-generator archive:
       <input>/imagenet_ai_<date>_<generator>/<train|val>/<ai|nature>/<image>
4. A split folder whose parent is a generator folder:
       <input>/<ai|nature>/<image>    where <input> is named train or val

Labels are taken only from the nearest ``ai`` or ``nature`` directory. Generator
identity is taken from the folder immediately above ``train``/``val`` (or from the
input directory when the input itself is a generator folder). Anything else is
rejected rather than guessed.

Grouping limitation
-------------------
GenImage's directory layout does not expose which real image or prompt produced a
fake image. This script therefore sets ``group_id`` to a SHA-256 identity of exact
file bytes. That groups only byte-identical files. It does NOT detect near-duplicates
or shared source images. Exact duplicate files are removed from the manifest, so the
remaining group IDs must not be presented as source-level groupings.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_FOLDERS = {"ai": 1, "nature": 0}
SPLIT_FOLDERS = {"train", "val"}

CANONICAL_GENERATORS = (
    "midjourney",
    "sd_1_4",
    "sd_1_5",
    "adm",
    "glide",
    "wukong",
    "vqdm",
    "biggan",
)

REQUIRED_COLUMNS = [
    "dataset",
    "file_path",
    "label",
    "generator",
    "source",
    "group_id",
]

MANIFEST_COLUMNS = REQUIRED_COLUMNS + [
    "source_split",
    "format",
    "grouping_basis",
    "grouping_limitation",
]

_ARCHIVE_RE = re.compile(r"^imagenet_ai_\d{4}_(.+)$", re.IGNORECASE)

# Keys are normalized names (lowercase, punctuation converted to spaces).
_GENERATOR_ALIASES = {
    "midjourney": "midjourney",
    "mj": "midjourney",
    "sd 1 4": "sd_1_4",
    "sd v1 4": "sd_1_4",
    "sd1 4": "sd_1_4",
    "sd14": "sd_1_4",
    "sdv4": "sd_1_4",
    "sdv1 4": "sd_1_4",
    "stable diffusion 1 4": "sd_1_4",
    "stable diffusion v1 4": "sd_1_4",
    "stable diffusion v 1 4": "sd_1_4",
    "sd 1 5": "sd_1_5",
    "sd v1 5": "sd_1_5",
    "sd1 5": "sd_1_5",
    "sd15": "sd_1_5",
    "sdv5": "sd_1_5",
    "sdv1 5": "sd_1_5",
    "stable diffusion 1 5": "sd_1_5",
    "stable diffusion v1 5": "sd_1_5",
    "stable diffusion v 1 5": "sd_1_5",
    "adm": "adm",
    "glide": "glide",
    "wukong": "wukong",
    "vqdm": "vqdm",
    "vq diffusion": "vqdm",
    "biggan": "biggan",
    "big gan": "biggan",
}


class ManifestError(ValueError):
    """Base error for manifest-generation problems."""


class GenImageLayoutError(ManifestError):
    """Raised when a path does not match a supported GenImage layout."""


class EmptyScanError(ManifestError):
    """Raised when a scan finds no usable images."""


@dataclass
class ScanStats:
    candidate_files: int = 0
    duplicate_paths: int = 0
    duplicate_contents: int = 0
    invalid_images: int = 0


def _normalize_token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_\-\.]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def canonical_generator(folder_name: str) -> str:
    """Return the canonical GenImage generator name for a folder/archive name."""
    raw_name = folder_name.strip()

    archive_match = _ARCHIVE_RE.match(raw_name)
    if archive_match:
        raw_name = archive_match.group(1)

    normalized = _normalize_token(raw_name)

    if normalized in {"imagenet ai", "imagenet-ai"}:
        raise GenImageLayoutError(
            "The merged 'imagenet_ai' layout does not contain enough information to "
            "preserve generator identity. Build one manifest per generator archive "
            "before merging, or provide an explicit generator mapping."
        )

    canonical = _GENERATOR_ALIASES.get(normalized)
    if canonical:
        return canonical

    expected = ", ".join(CANONICAL_GENERATORS)
    raise GenImageLayoutError(
        f"Unrecognized GenImage generator folder '{folder_name}'. "
        f"Expected one of: {expected}, or the matching "
        "'imagenet_ai_<date>_<generator>' archive folder."
    )


def verify_image(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
            return img.size[0] >= 32 and img.size[1] >= 32
    except Exception:
        return False


def _file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_genimage_record(
    file_path: Path,
    input_dir: Path,
    data_root: Path,
    dataset: str,
) -> dict:
    """Parse one image path using only the path relative to the input root."""
    resolved_file = file_path.resolve()
    input_root = input_dir.resolve()

    try:
        relative_to_input = resolved_file.relative_to(input_root)
    except ValueError as exc:
        raise GenImageLayoutError(
            f"Image '{file_path}' is outside the input directory '{input_dir}'."
        ) from exc

    parts = relative_to_input.parts

    # Find the nearest ai/nature ancestor. This deliberately ignores any
    # misleading parent-directory names above the dataset root.
    label_index: Optional[int] = None
    for index in range(len(parts) - 2, -1, -1):
        if parts[index].lower() in LABEL_FOLDERS:
            label_index = index
            break

    if label_index is None:
        raise GenImageLayoutError(
            f"Unrecognized GenImage layout for '{relative_to_input.as_posix()}'. "
            "Expected '<generator>/<train|val>/<ai|nature>/<image>' or the same "
            "shape with the generator/split as the input directory."
        )

    label_folder = parts[label_index].lower()
    label = LABEL_FOLDERS[label_folder]

    if label_index == 0:
        # The input directory is the split folder, e.g. Midjourney/train/ai/...
        split_name = input_dir.name.lower()
        if split_name not in SPLIT_FOLDERS:
            raise GenImageLayoutError(
                f"Expected the input directory to be named 'train' or 'val' when "
                f"it directly contains '{label_folder}', found '{input_dir.name}'."
            )
        generator_folder = input_dir.parent.name
    else:
        split_name = parts[label_index - 1].lower()
        if split_name not in SPLIT_FOLDERS:
            raise GenImageLayoutError(
                f"Expected 'train' or 'val' immediately above the '{label_folder}' "
                f"folder, found '{parts[label_index - 1]}' in "
                f"'{relative_to_input.as_posix()}'."
            )

        if label_index >= 2:
            generator_folder = parts[label_index - 2]
        else:
            # The input directory is the generator folder.
            generator_folder = input_dir.name

    # Validate the folder even for real images, but do not claim that ImageNet
    # images were produced by the surrounding GenImage generator.
    canonical = canonical_generator(generator_folder)
    generator = "real" if label == 0 else canonical

    try:
        relative_to_data_root = resolved_file.relative_to(data_root.resolve())
    except ValueError as exc:
        raise GenImageLayoutError(
            f"Image '{resolved_file}' is outside the manifest data root "
            f"'{data_root}'. Choose a data root that contains the input directory."
        ) from exc

    source = "imagenet" if label == 0 else f"{dataset}/{generator_folder}"

    return {
        "dataset": dataset,
        "file_path": relative_to_data_root.as_posix(),
        "label": label,
        "generator": generator,
        "source": source,
        "source_split": split_name,
        "format": resolved_file.suffix.lower().lstrip("."),
    }


def scan_image_directory(
    input_dir: Path,
    dataset: str = "genimage",
    data_root: Optional[Path] = None,
    verify: bool = False,
    stats: Optional[ScanStats] = None,
) -> list[dict]:
    """Build manifest records for a GenImage directory tree.

    ``file_path`` is stored relative to ``data_root``. When ``data_root`` is not
    supplied, paths are relative to ``input_dir``.
    """
    input_dir = Path(input_dir)
    stats = stats or ScanStats()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset or ""):
        raise ManifestError(
            "Dataset name must contain only letters, numbers, dots, underscores, "
            "and hyphens."
        )

    if not input_dir.exists():
        raise ManifestError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ManifestError(f"Input path is not a directory: {input_dir}")

    data_root_path = Path(data_root) if data_root is not None else input_dir

    candidate_files = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix(),
    )
    stats.candidate_files = len(candidate_files)

    if not candidate_files:
        raise EmptyScanError(
            f"No image files found under '{input_dir}'. Supported extensions: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}."
        )

    records: list[dict] = []
    seen_paths: set[Path] = set()
    seen_hashes: dict[str, dict] = {}

    for file_path in candidate_files:
        resolved_path = file_path.resolve()

        if resolved_path in seen_paths:
            stats.duplicate_paths += 1
            continue
        seen_paths.add(resolved_path)

        record = _parse_genimage_record(
            file_path=file_path,
            input_dir=input_dir,
            data_root=data_root_path,
            dataset=dataset,
        )

        if verify and not verify_image(file_path):
            stats.invalid_images += 1
            continue

        digest = _file_sha256(file_path)

        if digest in seen_hashes:
            previous = seen_hashes[digest]
            if previous["label"] != record["label"]:
                raise ManifestError(
                    "Identical file content appears with conflicting labels. "
                    f"First: '{previous['file_path']}' (label {previous['label']}); "
                    f"duplicate: '{record['file_path']}' (label {record['label']}). "
                    "Resolve the dataset labelling conflict before splitting."
                )
            stats.duplicate_contents += 1
            continue

        seen_hashes[digest] = record
        record["group_id"] = f"sha256:{digest}"
        record["grouping_basis"] = "sha256_exact_content"
        record["grouping_limitation"] = (
            "near_duplicates_and_shared_source_images_not_detected"
        )
        records.append(record)

    if not records:
        raise EmptyScanError(
            f"No usable images remained after scanning '{input_dir}'. "
            f"Invalid images skipped: {stats.invalid_images}."
        )

    return records


def write_manifest(records: list[dict], output_path: Path) -> pd.DataFrame:
    if not records:
        raise EmptyScanError("No manifest rows to write.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    df.to_csv(output_path, index=False)
    return df


def _print_value_counts(df: pd.DataFrame, column: str, title: str) -> None:
    print(f"\n{title}:")
    counts = df[column].value_counts(dropna=False).sort_index()
    if counts.empty:
        print("  (none)")
        return
    for value, count in counts.items():
        print(f"  {value}: {int(count)}")


def _print_label_counts(df: pd.DataFrame) -> None:
    print("\nLabel counts (0=real, 1=fake):")
    counts = df["label"].value_counts().sort_index()
    if counts.empty:
        print("  (none)")
        return
    for label, count in counts.items():
        name = {0: "real", 1: "fake"}.get(int(label), "unexpected")
        print(f"  {int(label)} ({name}): {int(count)}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a portable GenImage manifest from official directory layouts."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root containing GenImage generator folders, or one generator/split folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/images",
        help="Directory in which to write the manifest.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="genimage",
        help="Dataset name stored in the manifest (default: genimage).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Manifest filename. Defaults to <dataset>_manifest.csv.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Directory that file_path values are stored relative to. "
            "Defaults to the input directory."
        ),
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="genimage",
        choices=["genimage"],
        help="Directory layout to parse. Only the official GenImage layout is supported.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that each image can be fully decoded before including it.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_dir)
    data_root = Path(args.data_root) if args.data_root else None
    stats = ScanStats()

    try:
        records = scan_image_directory(
            input_dir=input_path,
            dataset=args.dataset,
            data_root=data_root,
            verify=args.verify,
            stats=stats,
        )
        output_name = args.output_name or f"{args.dataset}_manifest.csv"
        output_path = Path(args.output_dir) / output_name
        df = write_manifest(records, output_path)
    except ManifestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\nScanned: {input_path.resolve()}")
    print(f"Path base for file_path: {(data_root or input_path).resolve()}")
    print(f"Candidate image files: {stats.candidate_files}")
    print(f"Duplicate paths skipped: {stats.duplicate_paths}")
    print(f"Duplicate contents skipped: {stats.duplicate_contents}")
    print(f"Invalid images skipped: {stats.invalid_images}")
    print(f"Manifest rows written: {len(df)}")
    print(f"Manifest: {output_path.resolve()}")

    _print_value_counts(df, "dataset", "Dataset counts")
    _print_label_counts(df)
    _print_value_counts(df, "generator", "Generator counts")
    _print_value_counts(df, "source_split", "Official source_split counts")
    _print_value_counts(df, "source", "Source counts")

    print("\nGROUPING LIMITATION:")
    print("  group_id uses SHA-256 of exact file bytes only.")
    print("  It does NOT detect near-duplicates or shared source images.")
    print("  GenImage's folder layout does not expose the real-image/prompt mapping here.")
    print("  Do not treat these group IDs as source-level groupings.")


if __name__ == "__main__":
    main()
