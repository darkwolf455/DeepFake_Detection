# VeriSense: Multimodal Deepfake Detection System
## Production PyTorch System for Images, Videos, and Audio

VeriSense is an end-to-end multimodal deepfake detection system designed for high detection accuracy across multiple manipulation modalities:
1. **Image Branch**: Frozen CLIP ViT-L/14 visual backbone with a trainable non-linear classification head or optional PEFT/LoRA fine-tuning.
2. **Video Branch**: MediaPipe / OpenCV face extraction with margins -> CLIP / EfficientNet spatial embeddings -> Temporal Transformer or BiLSTM sequence model for temporal artifact and flickers detection.
3. **Audio Branch**: Wav2Vec2-XLS-R-300M (or WavLM-Large) self-supervised speech representation with layer-weighted summation and attentive multi-head pooling.
4. **Multimodal Late Fusion & Lip-Sync**: AV sync consistency scoring via SyncNet cross-modal embedding similarity + Platt temperature scaling and isotonic calibrated late fusion classifier with configurable "inconclusive" certainty margins.
5. **Explainability**: Spatial Grad-CAM overlays for image and video frames, plus second-by-second timeline risk charts for video and audio.

---

### Folder Structure
```
verisense/
├── README.md                      # Complete setup, datasets, and execution guide
├── LIMITATIONS.md                 # Failure modes, robustness boundaries, and ethics
├── requirements.txt               # Python package dependencies (PyTorch 2.x, transformers, etc.)
├── environment.yml                # Conda environment definition
├── docker-compose.yml             # Docker compose configuration
├── configs/
│   ├── image.yaml                 # Image model, augmentations, and training hyperparameters
│   ├── video.yaml                 # Video face extraction, temporal transformer config
│   ├── audio.yaml                 # Wav2Vec2/WavLM, RawBoost, sample rate, and training config
│   └── fusion.yaml                # Late fusion, isotonic/Platt calibration, confidence bands
├── data/
│   ├── README.md                  # Layout guide and symlink instructions
│   ├── raw/                       # Downloaded raw datasets
│   ├── processed/                 # Extracted faces, audio chunks, and precomputed features
│   └── splits/                    # Train/val/test CSV splits partitioned by identity/video
├── scripts/
│   ├── prepare_image_data.py      # Standardize images and balance generator folders
│   ├── prepare_video_data.py      # Extract faces and cache per-frame CLIP embeddings (.npy)
│   ├── prepare_audio_data.py      # Normalize 16kHz mono audio and crop 4s segments
│   └── make_splits.py             # Prevent identity leakage across splits
├── src/
│   ├── common/
│   │   ├── config.py              # YAML loader with dot-access & validation
│   │   ├── utils.py               # Seed, device selection, AMP, checkpointing, early stopping
│   │   └── metrics.py             # Accuracy, AUC-ROC, Average Precision, EER, minDCF, ECE
│   ├── image/
│   │   ├── dataset.py             # Multi-source dataset with Albumentations social compression
│   │   ├── model.py               # CLIP ViT-L/14 + Trainable Head + LoRA option
│   │   ├── train.py               # AMP training with CosineAnnealingLR and checkpointing
│   │   └── evaluate.py            # Test evaluation, ROC curve, and compression robustness test
│   ├── video/
│   │   ├── face_extractor.py      # Fast MediaPipe / OpenCV face detector with margin
│   │   ├── dataset.py             # Sequence dataset reading pre-extracted embeddings or video
│   │   ├── model.py               # Temporal Transformer Encoder / BiLSTM over frame tokens
│   │   ├── train.py               # Sequence training with focal/cross-entropy loss
│   │   └── evaluate.py            # Video-level AUC, intra- and cross-dataset testing
│   ├── audio/
│   │   ├── dataset.py             # 16kHz waveform loader with RawBoost / codec perturbations
│   │   ├── model.py               # Wav2Vec2 XLS-R layer-weighted sum + attentive pooling head
│   │   ├── train.py               # Audio training with EER validation tracking
│   │   └── evaluate.py            # ASVspoof evaluation, EER, minDCF, and codec robustness
│   ├── fusion/
│   │   ├── lipsync.py             # Mouth ROI to audio speech feature cross-correlation
│   │   ├── build_features.py      # Multi-branch feature generation for multimodal datasets
│   │   └── train_fusion.py        # Calibrated late fusion model with missing modality imputation
│   └── explain/
│       └── gradcam.py             # ViT / CNN Grad-CAM saliency map generator
├── backend/
│   ├── app.py                     # FastAPI application with lifecycle & CORS
│   ├── settings.py                # Environment configurations and checkpoint paths
│   ├── schemas.py                 # Pydantic request/response data contracts
│   ├── preprocessing.py           # Magic-byte MIME detection and media normalization
│   ├── inference.py               # Pipeline orchestrator with concurrency semaphore
│   └── Dockerfile                 # CUDA 12.1 runtime container
├── checkpoints/
│   └── README.md                  # Instructions for placing model weights (.pt / .safetensors)
└── tests/
    ├── test_metrics.py            # Unit tests for EER, AUC, ECE calculations
    ├── test_preprocessing.py      # Tests for MIME validation and face margins
    ├── test_datasets.py           # Dataset slicing and augmentation unit tests
    └── test_api.py                # FastAPI endpoint tests using synthetic media
```

---

### System Requirements & Hardware Targets
- **OS**: Linux (Ubuntu 20.04/22.04 recommended) or Windows 10/11 (with CUDA Toolkit 11.8 or 12.x).
- **Python**: 3.10 or 3.11.
- **GPU**: 1x NVIDIA GPU with 8GB to 12GB VRAM (e.g. RTX 3060, 3070, 4060, 4070, T4, A10).
- **RAM**: 16GB minimum (32GB recommended for caching audio/video features).
- **Disk**: 150GB+ SSD recommended for dataset storage and pre-extracted frame features.
- **CPU Fallback**: Full code supports `device = 'cpu'` automatically if CUDA is unavailable.

---

### Recommended Dataset Downloads & Licenses
Please ensure you comply with the respective academic research licenses and acceptable use agreements before downloading:

| Dataset | Modality | Primary Use | Approx Size | Download Link / Access |
|---|---|---|---|---|
| **FaceForensics++** | Video, Image | DeepFakes, Face2Face, FaceSwap, NeuralTextures | ~80 GB (c23) | [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) (Google Form request) |
| **Celeb-DF v2** | Video | High-quality cross-dataset generalization | ~35 GB | [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **GenImage** | Image | 8 generators (Midjourney, SD, DALL-E, etc.) | ~90 GB | [github.com/GenImage-Dataset/GenImage](https://github.com/GenImage-Dataset/GenImage) |
| **ASVspoof 2019 LA** | Audio | Synthetic speech & voice cloning | ~15 GB | [datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336) |
| **In-the-Wild Audio** | Audio | Real-world deepfake voice benchmark | ~4 GB | [deepfake-total.com/in_the_wild](https://deepfake-total.com/in_the_wild) |
| **FakeAVCeleb** | Multimodal | Audio-visual deepfakes and lip-sync swaps | ~32 GB | [github.com/DASH-Lab/FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) (Application required) |

#### Quick-Start Small Subset Recommendation
If you have limited bandwidth or disk space:
1. **Images**: Download GenImage (Stable Diffusion v1.4 vs Real Imagenet subset, ~5,000 real / 5,000 fake).
2. **Videos**: Download FaceForensics++ c23 (Deepfakes + Real Actors subset, 100 real / 100 fake videos).
3. **Audio**: Download ASVspoof 2019 LA train partition (~2,500 real / 2,500 spoof files).
4. Run scripts with the `--debug` flag to run a complete smoke test in under 5 minutes.

---

### Step-by-Step Training Execution Order

#### Step 0: Environment Setup
```bash
# Option A: Conda
conda env create -f environment.yml
conda activate verisense

# Option B: Venv & pip
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 1: Data Preparation & Leak-Free Splits
```bash
# 1. Image dataset preparation (normalizes real/fake labels and checks corrupt images)
python scripts/prepare_image_data.py --input-dir data/raw/genimage --output-dir data/processed/images

# 2. Extract faces and precompute CLIP frame tokens for videos
python scripts/prepare_video_data.py --video-dir data/raw/faceforensics --output-dir data/processed/video_cache --fps 2 --max-frames 32

# 3. Audio preprocessing (resample to 16kHz mono, save chunks)
python scripts/prepare_audio_data.py --input-dir data/raw/asvspoof2019 --output-dir data/processed/audio --duration 4.0

# 4. Generate splits partitioned by speaker/subject ID to strictly prevent data leakage
python scripts/make_splits.py --processed-dir data/processed --output-dir data/splits
```

#### Step 2: Smoke-Test / Quick Verification (--debug)
Before initiating multi-hour runs, verify every component on a mini-batch:
```bash
python src/image/train.py --config configs/image.yaml --debug
python src/video/train.py --config configs/video.yaml --debug
python src/audio/train.py --config configs/audio.yaml --debug
python src/fusion/train_fusion.py --config configs/fusion.yaml --debug
```

#### Step 3: Full Model Training
```bash
# 1. Train Image Branch (CLIP ViT-L/14 + Trainable MLP / LoRA)
python src/image/train.py --config configs/image.yaml

# 2. Train Video Branch (Temporal Transformer over cached frame tokens)
python src/video/train.py --config configs/video.yaml

# 3. Train Audio Branch (Wav2Vec2 XLS-R layer aggregation + Attentive Pooling)
python src/audio/train.py --config configs/audio.yaml

# 4. Extract Multimodal Branch Predictions on Held-Out Validation/Test Sets
python src/fusion/build_features.py --config configs/fusion.yaml

# 5. Train & Calibrate Late-Fusion Classifier (Logistic Regression / MLP + Temperature Scaling)
python src/fusion/train_fusion.py --config configs/fusion.yaml
```

#### Step 4: Comprehensive Evaluation
```bash
python src/image/evaluate.py --config configs/image.yaml --checkpoint checkpoints/image_best.pt
python src/video/evaluate.py --config configs/video.yaml --checkpoint checkpoints/video_best.pt
python src/audio/evaluate.py --config configs/audio.yaml --checkpoint checkpoints/audio_best.pt
```

#### Step 5: Run Automated Unit Tests
```bash
pytest tests/ -v
```

#### Step 6: Launch Backend API Server
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# View interactive Swagger API documentation at: http://localhost:8000/docs
```

---

### Troubleshooting & Common Gotchas
- **Windows `num_workers > 0` Crash**: On Windows, PyTorch `DataLoader(num_workers > 0)` can fail with `BrokenPipeError` or multiprocessing pickle errors. Set `num_workers: 0` in YAML or verify the `if __name__ == '__main__':` guard.
- **CUDA Out-of-Memory (OOM)**:
  1. Reduce `batch_size` in the config (e.g. from 32 to 16 or 8).
  2. Increase `gradient_accumulation_steps` (e.g., from 1 to 4) to maintain an effective batch size.
  3. Ensure `use_amp: true` is enabled in your YAML to use Float16/BFloat16 mixed precision.
- **FFmpeg Not Found**: Install ffmpeg via `sudo apt install ffmpeg` on Linux, or `choco install ffmpeg` / download binary on Windows and add to system `PATH`.
- **Missing Checkpoint Behavior**: If a model checkpoint is absent, the backend server automatically runs in graceful fallback mode: the missing branch is reported under `warnings`, and the remaining active branches continue serving predictions.
