# VeriSense Data Directory Layout

This directory stores raw datasets, processed manifests, and train/validation/test
splits. This document describes the image pipeline in detail and notes where the
video and audio branches share the same structure.

## Approved datasets

Only these four datasets are approved for the project. Do not add others without
approval, and do not merge unrelated datasets into one manifest.

| Dataset | Modality | Primary purpose |
|---|---|---|
| **GenImage** | Image | Primary image benchmark. Real ImageNet photographs plus synthetic images from eight generators. Used for cross-generator generalization experiments. |
| **FaceForensics++** | Video (frames can feed the image branch) | Face-manipulation detection: DeepFakes, Face2Face, FaceSwap, and NeuralTextures. |
| **Celeb-DF v2** | Video | High-quality celebrity deepfakes. Mainly for cross-dataset generalization. |
| **FakeAVCeleb** | Multimodal (video + audio) | Audio-visual deepfakes and lip-sync manipulation. Mainly for multimodal fusion. |

ASVspoof 2019 is used by the separate audio branch. It is not part of the
approved image-dataset set above.

## Directory layout

```text
data/
├── raw/
│   ├── genimage/                  # Official per-generator GenImage folders
│   ├── faceforensics/             # FaceForensics++ videos
│   ├── celeb_df_v2/               # Celeb-DF v2 videos
│   ├── fakeavceleb/               # FakeAVCeleb videos/audio
│   └── asvspoof2019/              # Audio branch only
├── processed/
│   ├── images/
│   │   └── genimage_manifest.csv  # Written by prepare_image_data.py
│   ├── video_cache/               # Precomputed video features and face boxes
│   ├── audio/                     # Resampled audio chunks
│   └── multimodal_features.csv    # Branch predictions for late fusion
└── splits/
    ├── image_train.csv
    ├── image_val.csv
    ├── image_test.csv
    ├── image_split_metadata.json  # Protocol, seed, counts, warnings, limitations
    ├── video_train.csv
    ├── video_val.csv
    ├── video_test.csv
    ├── audio_train.csv
    ├── audio_val.csv
    ├── audio_test.csv
    ├── fusion_train.csv
    ├── fusion_val.csv
    └── fusion_test.csv
```

The image split CSVs contain the full manifest schema below. They do **not** add
an extra row-order column.

## GenImage directory layout

GenImage is organised by generator, then by the official split, then by label:

```text
data/raw/genimage/
├── Midjourney/
│   ├── train/
│   │   ├── ai/          # label = 1 (fake)
│   │   └── nature/      # label = 0 (real)
│   └── val/
│       ├── ai/
│       └── nature/
├── Stable Diffusion V1.4/
├── Stable Diffusion V1.5/
├── ADM/
├── GLIDE/
├── Wukong/
├── VQDM/
└── BigGAN/
```

Extracted per-generator archives are also supported:

```text
data/raw/genimage/imagenet_ai_0419_sdv4/{train,val}/{ai,nature}/...
```

`prepare_image_data.py` accepts four shapes:

1. A root containing generator folders:
   `<input>/<Generator>/<train|val>/<ai|nature>/<image>`
2. A single generator folder:
   `<input>/<train|val>/<ai|nature>/<image>`
3. An extracted per-generator archive:
   `<input>/imagenet_ai_<date>_<generator>/<train|val>/<ai|nature>/<image>`
4. A split folder whose parent is a generator folder:
   `<input>/<ai|nature>/<image>`, where `<input>` is named `train` or `val`

Labels come only from the nearest `ai` or `nature` folder. A parent directory
named `ai`, `fake`, or `real` elsewhere in the path does not change the label.

The merged `imagenet_ai` layout is rejected because it no longer preserves which
generator produced each fake image.

Canonical generator IDs stored in the manifest:

```text
midjourney, sd_1_4, sd_1_5, adm, glide, wukong, vqdm, biggan
```

Real images use `generator = "real"`. Real images are never treated as outputs of
a fake generator.

## Manifest columns and label conventions

`data/processed/images/genimage_manifest.csv` has these columns, in this order:

| Column | Meaning |
|---|---|
| `dataset` | Dataset name, for example `genimage`. |
| `file_path` | Portable path relative to the builder's `--data-root` (default: the input directory). The builder writes relative paths; the loader resolves them against `data.data_root`. |
| `label` | `0` = real, `1` = fake. |
| `generator` | Canonical fake-generator ID for fakes; `real` for real images. |
| `source` | `imagenet` for real images; `<dataset>/<generator_folder>` for fakes. |
| `group_id` | `sha256:<digest>` identity of exact file bytes. |
| `source_split` | Official GenImage split: `train` or `val`. |
| `format` | Lowercase extension without the dot, for example `jpg`. |
| `grouping_basis` | `sha256_exact_content`. |
| `grouping_limitation` | `near_duplicates_and_shared_source_images_not_detected`. |

Exact duplicate paths and byte-identical files are removed during manifest
generation. Identical bytes with conflicting labels cause a hard error.

## Generating the image manifest

From the `verisense/` project root:

```bash
python scripts/prepare_image_data.py \
  --input-dir data/raw/genimage \
  --output-dir data/processed/images \
  --data-root .
```

`--data-root .` makes `file_path` values start with `data/raw/genimage/...`,
relative to the `verisense/` directory. The image config's `data.data_root` must
identify the same directory so the loader can resolve those paths. See
[Path resolution](#path-resolution) below.

Useful options:

```bash
--dataset genimage          # stored in the dataset column
--output-name genimage_manifest.csv
--verify                    # fully decode and reject corrupt/small images
```

## Generating image splits

```bash
python scripts/make_splits.py \
  --processed-dir data/processed \
  --output-dir data/splits \
  --protocol cross_generator \
  --val-generators sd_1_4 \
  --test-generators biggan \
  --seed 42
```

`make_splits.py` uses exactly one `*_manifest.csv` found under
`data/processed/images`. If several manifests exist, pass `--manifest` explicitly.
It does not merge unrelated datasets or manifests.

Generator arguments use the values stored in the manifest's `generator` column,
not the original folder names. For example, use `sd_1_4`, not
`Stable Diffusion V1.4`.

### `cross_generator` (default)

Entire fake generators are held out. A fake generator used in training never
appears in validation or test.

```text
Train fake generators:      all generators except the held-out ones
Validation fake generators: --val-generators
Test fake generators:       --test-generators
```

- Validation and test generators must be distinct.
- Held-out generators contribute **all** their rows, from both official `train`
  and `val` folders, to their held-out split.
- Training generators contribute both their official `train` and `val` rows to
  training. This is an explicit cross-generator choice, not the official GenImage
  benchmark protocol.
- Real images are pooled and split separately, group-wise, across train/val/test.
- `source_split` is preserved in the output as metadata, but generator holdout
  controls assignment.
- At least three fake generators are required. If they are not specified, the
  validation and test generators are chosen deterministically from the seed.

### `in_distribution`

The same generators may appear in every split, but `group_id` values must remain
disjoint.

```bash
python scripts/make_splits.py --protocol in_distribution --seed 42
```

With the default `--source-split-policy auto`, this becomes `respect`:

- Official `train` rows are split group-wise into train and validation.
- Official `val` rows become test.
- A group spanning both official pools causes a hard error.

To pool official rows deliberately instead:

```bash
python scripts/make_splits.py --protocol in_distribution --source-split-policy ignore
```

`ignore` is explicit. The splitter never silently falls back to row-level random
splitting when grouping data is missing or invalid.

### Output files

- `data/splits/image_train.csv`
- `data/splits/image_val.csv`
- `data/splits/image_test.csv`
- `data/splits/image_split_metadata.json`

The metadata file records the protocol, seed, selected generators, split counts,
grouping basis, grouping limitation, and warnings.

## Grouping limitation

`group_id` currently represents **SHA-256 exact-file-content identity only**.

- It catches byte-identical files.
- It does **not** detect near-duplicates.
- It does **not** detect different images derived from the same original source,
  prompt, or subject.

The current splits are therefore **exact-content-group-disjoint**, not fully
leakage-free and not source-disjoint. `sha256_exact_content` grouping cannot
prevent source-image-level leakage. Do not describe these splits as
source-level leakage prevention.

Group overlap is validated, and the grouping basis and limitation are reported
in both the console output and `image_split_metadata.json`. The split is only as
good as the `group_id` values supplied by the manifest.

## Path resolution

The manifest stores portable `file_path` values, and the image config supplies
the base directory used to resolve them:

```yaml
data:
  # Must match the --data-root used by prepare_image_data.py.
  data_root: "."
  train_split_csv: "data/splits/image_train.csv"
  val_split_csv: "data/splits/image_val.csv"
  test_split_csv: "data/splits/image_test.csv"
```

`data.data_root` is the base directory for relative `file_path` values.
`src/image/train.py` and `src/image/evaluate.py` read it and pass it explicitly
to `DeepfakeImageDataset`. If it is missing or empty, both scripts raise an error
instead of guessing a root. `scripts/prepare_image_data.py` and
`scripts/make_splits.py` do not read this config field.

### Resolution rules

`DeepfakeImageDataset` resolves every `file_path` once, when the dataset is
constructed:

| Manifest `file_path` | Behavior |
|---|---|
| Relative, for example `data/raw/genimage/.../0001.jpg` | Joined to `data_root`, then resolved to an absolute path. |
| Absolute, for example `/mnt/data/genimage/.../0001.jpg` | Kept absolute. The dataset loader does not need `data_root` for absolute-only manifests, but `train.py` and `evaluate.py` still require `data.data_root` to be set. |
| Relative, but no `data_root` supplied | `ValueError`; the loader never guesses a root from the working directory. |
| Relative, but `data_root` does not exist | `FileNotFoundError`. |
| Relative, but `data_root` is a file | `NotADirectoryError`. |
| Any, but the image is missing | `ImageReadError` with the manifest path, resolved path, and `data_root`. |
| Any, but the image is corrupt or unreadable | `ImageReadError`; no black or dummy image is substituted. |

Absolute paths are resolved with `Path.resolve()`, so `..` segments and symlinks
are normalized. The manifest builder always writes relative paths; absolute
paths are supported for manifests produced elsewhere.

`data_root: "."` is relative to the **process working directory at dataset
construction time**. Launched from `verisense/`, it means the `verisense/`
directory. Because `train.py` and `evaluate.py` construct the dataset during
startup, the working directory at launch is what matters. The resolved absolute
paths are stored in the dataset, so changing the working directory afterward
does not change which files are opened. For jobs launched from another
directory, use an absolute `data_root` in the config.

The split CSV paths (`train_split_csv`, `val_split_csv`, `test_split_csv`) are
separate: they are opened relative to the process working directory and are not
joined to `data_root`.

### Keeping the manifest builder and loader in agreement

The manifest builder's `--data-root` and the image config's `data.data_root` must
identify the same base directory:

- `scripts/prepare_image_data.py --data-root X` stores each `file_path` relative
  to `X`. Images outside `X` are rejected.
- `data.data_root: Y` makes the loader open `Y / file_path`.
- If `X` and `Y` differ, paths resolve to the wrong place and loading fails with
  `ImageReadError` (missing file).
- `scripts/make_splits.py` preserves `file_path` unchanged; it does not rewrite
  paths or apply `data_root`.

For example, this command stores paths such as
`data/raw/genimage/Midjourney/train/ai/0001.jpg`:

```bash
python scripts/prepare_image_data.py \
  --input-dir data/raw/genimage \
  --output-dir data/processed/images \
  --data-root .
```

The matching config uses `data_root: "."` from the same directory. Building with
`--data-root /absolute/path/to/verisense` instead requires
`data_root: "/absolute/path/to/verisense"`.

### Recommended workflow

Run every image command from the `verisense/` project root so the relative config
paths and `data_root: "."` all refer to the same directory.

1. Build the manifest:

```bash
cd verisense   # from the repository root
python scripts/prepare_image_data.py \
  --input-dir data/raw/genimage \
  --output-dir data/processed/images \
  --data-root .
```

2. Generate splits:

```bash
python scripts/make_splits.py \
  --processed-dir data/processed \
  --output-dir data/splits \
  --protocol cross_generator \
  --val-generators sd_1_4 \
  --test-generators biggan \
  --seed 42
```

3. Train:

```bash
python src/image/train.py --config configs/image.yaml
```

4. Evaluate a checkpoint:

```bash
python src/image/evaluate.py \
  --config configs/image.yaml \
  --checkpoint checkpoints/image_best.pt
```

Training also accepts `--debug` for a short smoke run:

```bash
python src/image/train.py --config configs/image.yaml --debug
```

## Important split rule

Never split frames from the same video or recordings from the same speaker across
training and test sets. For images, keep exact-content `group_id` values disjoint.
Run `python scripts/make_splits.py` rather than creating splits by hand.
