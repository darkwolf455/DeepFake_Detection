"""
VeriSense Backend Settings & Environment Configuration
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "VeriSense Backend"
    version: str = "1.0.0"
    debug: bool = False

    # Server & Concurrency
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_inferences: int = 1

    # Upload Limits (Bytes)
    max_image_size: int = 25 * 1024 * 1024    # 25 MB

    # Checkpoint Paths
    image_checkpoint: Path = Path(__file__).resolve().parents[1] / "checkpoints/image_head.pt"
    video_checkpoint: Path = Path(__file__).resolve().parents[1] / "checkpoints/video_best.pt"
    audio_checkpoint: Path = Path(__file__).resolve().parents[1] / "checkpoints/audio_head.pt"
    fusion_checkpoint: Path = Path(__file__).resolve().parents[1] / "checkpoints/lavdf/fusion_best.pt"

    # Calibration & Confidence Bands
    inconclusive_lower: float = 0.40
    inconclusive_upper: float = 0.65

    class Config:
        env_file = ".env"
        env_prefix = "VERISENSE_"


settings = Settings()
