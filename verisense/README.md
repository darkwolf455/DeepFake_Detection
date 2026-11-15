# VeriSense research code

This folder contains the Python API, compact model heads and reproducible training
scripts. The application uses local PyTorch inference; no paid API is required.
See the [project README](../README.md) for installation and startup.

## Implemented branches

| Branch | Backbone and classifier | Runtime scope |
| --- | --- | --- |
| Image | Frozen CLIP ViT-L/14 + face head | Exactly one detected face |
| Video | Normalized CLIP features + temporal Transformer | Up to 16 samples in the first 30 seconds |
| Audio | Frozen Wav2Vec2 Base + learned pooling/head | Central four seconds of a WAV/FLAC clip |
| Fusion | Adapted image/audio heads + original temporal model + logistic classifier | Paired video/audio up to 30 seconds; at least four usable face samples |

The standalone FF++/ASVspoof heads remain separate from LAV-DF adaptation.
Fusion shares their frozen backbones and verifies checkpoint identities, face
backend and speech preprocessing against training provenance before loading.
Unavailable requested models return 503. Unusable fusion evidence returns 422;
there is no automatic replacement verdict. Lip-sync and heatmaps are not implemented.

## Where things live

- `backend/`: FastAPI routes, upload validation, settings and inference.
- `src/image/`, `src/video/`, `src/audio/`: datasets, models, training and evaluation.
- `src/fusion/`: paired feature caching, grouped OOF fusion training and scoring.
- `src/common/`: configuration, metrics, seeds and checkpoint helpers.
- `scripts/`: data preparation and controlled LAV-DF adaptation/evaluation.
- `configs/`: actual training settings.
- `tests/`: model-loading, preprocessing, source separation and API checks.
- `checkpoints/`: original weights; `checkpoints/lavdf/`: evaluated fusion bundle.

## Training and evaluation

Follow [COLAB_DEMO.md](COLAB_DEMO.md) for the free-Colab workflow, including Drive
persistence, dataset access, saved splits, feature caches and restart-ready cells.
Generated update archives are preserved locally under `../artifacts/colab/`.
The imported model bundle is under `../artifacts/exports/`; it is excluded from Git.

Use `requirements-api.txt` for local inference. `requirements.txt` adds research
and testing dependencies. The portable `imageio-ffmpeg` decoder avoids a system
FFmpeg installation for the local fusion route.

Recorded scores and their limitations are in [LIMITATIONS.md](LIMITATIONS.md).
Related samples are correlated; small benchmark results do not establish broad
reliability. The frozen LAV-DF test cohort had previously been inspected for the
initial fusion baseline. Model scores are not calibrated probabilities.
