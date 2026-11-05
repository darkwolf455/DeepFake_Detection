"""
Image Branch Dataset with Albumentations social media compression augmentations.
"""

from pathlib import Path
from typing import Callable, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class ImageReadError(RuntimeError):
    """Raised when an image file is missing, unreadable, or corrupt."""


def _resolve_image_path(raw_path, data_root: Optional[Path], row_number: int) -> Path:
    """Resolve one manifest path without guessing a root from the CWD."""
    if pd.isna(raw_path):
        raise ValueError(f"Row {row_number} has a missing file_path value.")

    text = str(raw_path).strip()
    if not text:
        raise ValueError(f"Row {row_number} has an empty file_path value.")

    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()

    if data_root is None:
        raise ValueError(
            f"Row {row_number} has relative file_path '{text}', but data_root is "
            "not set. Set data.data_root in configs/image.yaml; the loader will "
            "not guess a root from the working directory."
        )

    return (data_root / path).resolve()


def build_jpeg_compression(quality_min: int, quality_max: int, p: float) -> A.ImageCompression:
    """Create an ImageCompression transform compatible with Albumentations 1.x/2.x."""
    try:
        return A.ImageCompression(quality_range=(quality_min, quality_max), p=p)
    except TypeError:
        return A.ImageCompression(quality_lower=quality_min, quality_upper=quality_max, p=p)


def get_image_transforms(config, is_training: bool = True) -> A.Compose:
    img_size = config.data.image_size if hasattr(config, "data") else 224

    if is_training:
        aug_cfg = config.augmentation if hasattr(config, "augmentation") else None
        jpeg_min = aug_cfg.jpeg_compression_min if aug_cfg else 30
        jpeg_max = aug_cfg.jpeg_compression_max if aug_cfg else 100
        blur_prob = aug_cfg.gaussian_blur_prob if aug_cfg else 0.3
        hflip_prob = aug_cfg.horizontal_flip_prob if aug_cfg else 0.5
        jitter_prob = aug_cfg.color_jitter_prob if aug_cfg else 0.3

        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=hflip_prob),
            build_jpeg_compression(jpeg_min, jpeg_max, p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=blur_prob),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=jitter_prob),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], # OpenAI CLIP standard normalization
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
            ToTensorV2(),
        ])


class DeepfakeImageDataset(Dataset):
    """
    Multimodal/multi-source image dataset supporting:
    - Standard real/fake folders
    - FaceForensics++ extracted frames
    - GenImage generator subdirectories (e.g. Midjourney, SD, DALL-E)
    """

    def __init__(
        self,
        csv_file: str | Path,
        transform: Optional[A.Compose] = None,
        debug_limit: Optional[int] = None,
        data_root: Optional[str | Path] = None,
    ):
        self.csv_file = Path(csv_file)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_file}")

        if data_root is not None and str(data_root).strip() == "":
            raise ValueError("data_root must not be empty when provided.")

        self.data_root = (
            Path(data_root).expanduser().resolve() if data_root is not None else None
        )

        self.df = pd.read_csv(self.csv_file)

        # Standardize column names
        col_map = {"path": "file_path", "image_path": "file_path", "filepath": "file_path"}
        self.df = self.df.rename(columns=col_map)

        if "file_path" not in self.df.columns:
            raise ValueError(
                "CSV must contain a 'file_path' column (or 'path'/'image_path'/'filepath')"
            )

        if "label" not in self.df.columns:
            raise ValueError("CSV must contain a 'label' column (0 for real, 1 for fake)")

        if debug_limit is not None and debug_limit > 0:
            self.df = self.df.head(debug_limit)

        raw_paths = self.df["file_path"].tolist()
        has_relative_paths = False
        for raw_path in raw_paths:
            if pd.isna(raw_path) or not str(raw_path).strip():
                continue
            if not Path(str(raw_path).strip()).expanduser().is_absolute():
                has_relative_paths = True
                break

        if has_relative_paths:
            if self.data_root is None:
                raise ValueError(
                    "Split CSV contains relative file_path values, but data_root was "
                    "not provided. Set data.data_root in configs/image.yaml; the "
                    "loader will not guess a root from the working directory."
                )
            if not self.data_root.exists():
                raise FileNotFoundError(
                    f"Configured data_root does not exist: {self.data_root}"
                )
            if not self.data_root.is_dir():
                raise NotADirectoryError(
                    f"Configured data_root is not a directory: {self.data_root}"
                )

        self._resolved_paths = [
            _resolve_image_path(raw_path, self.data_root, index)
            for index, raw_path in enumerate(raw_paths)
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        raw_path = row["file_path"]
        img_path = self._resolved_paths[idx]
        label = int(row["label"])
        generator = row["generator"] if "generator" in row else "unknown"

        if not img_path.exists():
            raise ImageReadError(
                f"Failed to read image '{img_path}': file not found. "
                f"Manifest file_path='{raw_path}', data_root='{self.data_root}'."
            )
        if not img_path.is_file():
            raise ImageReadError(
                f"Failed to read image '{img_path}': path is not a regular file. "
                f"Manifest file_path='{raw_path}', data_root='{self.data_root}'."
            )

        # Read image using OpenCV BGR -> RGB, with a strict PIL fallback.
        image = cv2.imread(str(img_path))
        if image is None or image.size == 0:
            try:
                with Image.open(img_path) as pil_img:
                    image = np.array(pil_img.convert("RGB"))
            except Exception as exc:
                raise ImageReadError(
                    f"Failed to read image '{img_path}'. The file may be unreadable "
                    f"or corrupt. Manifest file_path='{raw_path}', "
                    f"data_root='{self.data_root}'."
                ) from exc
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "generator": generator,
            "file_path": str(img_path),
        }

    def get_sampler(self) -> WeightedRandomSampler:
        """Constructs balanced sampler to counter class imbalance."""
        labels = self.df["label"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return sampler
