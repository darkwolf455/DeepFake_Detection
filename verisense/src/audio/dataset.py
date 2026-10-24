"""
Audio Branch Dataset: Loads 16kHz mono audio, segments into fixed 4s windows,
and applies RawBoost-style non-linear convolutional noise and codec simulation.
"""

from pathlib import Path
import random
from typing import Optional
import numpy as np
import pandas as pd
from scipy import signal
import soundfile as sf
import torch
from torch.utils.data import Dataset


def apply_rawboost_perturbation(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulates RawBoost linear and non-linear convolutive noise:
    random band-pass filtering + subtle harmonic distortion.
    """
    mode = random.choice(["linear_filter", "harmonic_distortion", "additive_noise"])

    if mode == "linear_filter":
        # Random low-pass or band-pass filter
        f_cut = random.uniform(3000, 7500)
        sos = signal.butter(4, f_cut, "lowpass", fs=sr, output="sos")
        waveform = signal.sosfilt(sos, waveform)
    elif mode == "harmonic_distortion":
        # Non-linear saturation / clipping simulation
        gain = random.uniform(1.2, 2.5)
        waveform = np.tanh(waveform * gain) / gain
    else:
        # Additive white gaussian noise (AWGN) with 15-30 dB SNR
        rms = np.sqrt(np.mean(waveform**2)) + 1e-8
        snr = random.uniform(15, 30)
        noise_std = rms / (10 ** (snr / 20.0))
        noise = np.random.normal(0, noise_std, waveform.shape)
        waveform = waveform + noise

    # Clip to valid audio bounds
    return np.clip(waveform, -1.0, 1.0)


class AudioDeepfakeDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        target_sr: int = 16000,
        duration: float = 4.0,
        is_training: bool = True,
        use_augmentation: bool = True,
        debug_limit: Optional[int] = None,
    ):
        self.csv_file = Path(csv_file)
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration)
        self.is_training = is_training
        self.use_augmentation = use_augmentation

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Audio split CSV not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)
        if debug_limit is not None and debug_limit > 0:
            self.df = self.df.head(debug_limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        file_path = row["processed_path"] if "processed_path" in row and pd.notna(row["processed_path"]) else row["filepath"]
        label = int(row["label"])
        speaker_id = str(row.get("speaker_id", "unknown"))

        try:
            waveform, sr = sf.read(file_path, dtype="float32")
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)  # Convert stereo to mono
        except Exception:
            waveform = np.zeros(self.target_samples, dtype=np.float32)
            sr = self.target_sr

        # Crop or pad to target length
        total_len = len(waveform)
        if total_len > self.target_samples:
            if self.is_training:
                # Random 4s crop during training
                start = random.randint(0, total_len - self.target_samples)
                waveform = waveform[start : start + self.target_samples]
            else:
                # Center crop during evaluation
                start = (total_len - self.target_samples) // 2
                waveform = waveform[start : start + self.target_samples]
        elif total_len < self.target_samples:
            pad_len = self.target_samples - total_len
            waveform = np.pad(waveform, (0, pad_len), mode="wrap")

        # Audio augmentations during training
        if self.is_training and self.use_augmentation and random.random() < 0.6:
            waveform = apply_rawboost_perturbation(waveform, sr=self.target_sr)

        # RMS Normalization
        rms = np.sqrt(np.mean(waveform**2))
        if rms > 1e-6:
            waveform = waveform * (0.1 / rms)

        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "speaker_id": speaker_id,
            "filepath": str(file_path),
        }
