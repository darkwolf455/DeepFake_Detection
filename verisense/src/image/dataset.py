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
            A.ImageCompression(quality_lower=jpeg_min, quality_upper=jpeg_max, p=0.6),
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
    ):
        self.csv_file = Path(csv_file)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)

        # Standardize column names
        col_map = {"path": "file_path", "image_path": "file_path"}
        self.df = self.df.rename(columns=col_map)

        if "label" not in self.df.columns:
            raise ValueError("CSV must contain a 'label' column (0 for real, 1 for fake)")

        if debug_limit is not None and debug_limit > 0:
            self.df = self.df.head(debug_limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        label = int(row["label"])
        generator = row["generator"] if "generator" in row else "unknown"

        # Read image using OpenCV BGR -> RGB
        image = cv2.imread(img_path)
        if image is None:
            # Fallback to PIL or blank canvas to prevent worker crashes
            try:
                pil_img = Image.open(img_path).convert("RGB")
                image = np.array(pil_img)
            except Exception:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
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
