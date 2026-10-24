# VeriSense Data Directory Layout

This directory stores raw datasets, preprocessed feature caches, and identity-partitioned splits.

```
data/
├── raw/
│   ├── faceforensics/       # e.g., downloaded c23 videos
│   ├── celeb_df_v2/         # Celeb-DF v2 videos
│   ├── genimage/            # GenImage subfolders (midjourney, sd_1_4, dalle, etc.)
│   └── asvspoof2019/        # ASVspoof 2019 LA flac audio files
├── processed/
│   ├── images/              # Cleaned & standardized image index
│   ├── video_cache/         # Precomputed per-frame embeddings (.npy) and face boxes
│   ├── audio/               # 16kHz resampled 4s audio chunks (.wav)
│   └── multimodal_features.csv  # Combined table of branch predictions for late-fusion
└── splits/
    ├── image_train.csv      # Columns: filepath, label (0=real, 1=fake), generator, split
    ├── image_val.csv
    ├── image_test.csv
    ├── video_train.csv      # Columns: video_id, embedding_path, label, num_frames
    ├── video_val.csv
    ├── video_test.csv
    ├── audio_train.csv      # Columns: audio_id, filepath, label, speaker_id
    ├── audio_val.csv
    ├── audio_test.csv
    ├── fusion_train.csv
    ├── fusion_val.csv
    └── fusion_test.csv
```

### Important Rule for Splits
Never split frames from the same video or audio recordings from the same speaker across training and testing sets.
Always run `python scripts/make_splits.py` to ensure leak-free identity-based partitioning.
