"""
VeriSense Backend Settings & Environment Configuration
"""

from pathlib import Path
from typing import List, Tuple
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "VeriSense Backend"
    version: str = "1.0.0"
    debug: bool = False

    # Server & Concurrency
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_inferences: int = 2  # GPU semaphore limit to prevent VRAM exhaustion
    request_timeout_seconds: int = 120

    # Upload Limits (Bytes)
    max_image_size: int = 25 * 1024 * 1024    # 25 MB
    max_video_size: int = 150 * 1024 * 1024   # 150 MB
    max_audio_size: int = 50 * 1024 * 1024    # 50 MB

    # Checkpoint Paths
    checkpoint_dir: Path = Path("checkpoints")
    image_checkpoint: Path = Path("checkpoints/image_best.pt")
    video_checkpoint: Path = Path("checkpoints/video_best.pt")
    audio_checkpoint: Path = Path("checkpoints/audio_best.pt")
    fusion_checkpoint: Path = Path("checkpoints/fusion_calibrated.pt")

    # Calibration & Confidence Bands
    inconclusive_lower: float = 0.40
    inconclusive_upper: float = 0.65

    # Storage for temporary Grad-CAM heatmaps
    heatmaps_dir: Path = Path("heatmaps_cache")

    class Config:
        env_file = ".env"
        env_prefix = "VERISENSE_"


settings = Settings()
