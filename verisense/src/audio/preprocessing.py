"""The same mono, resampling, cropping and normalization for training and inference."""
import math
import random
import subprocess

import numpy as np


def prepare_audio(waveform, sample_rate, target_sr=16000, duration=4.0, training=False, augment=None):
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all() or sample_rate <= 0:
        raise ValueError("Audio is empty or contains invalid samples.")
    if sample_rate != target_sr:
        from scipy.signal import resample_poly
        divisor = math.gcd(sample_rate, target_sr)
        waveform = resample_poly(waveform, target_sr // divisor, sample_rate // divisor)
    samples = int(target_sr * duration)
    if len(waveform) > samples:
        start = random.randint(0, len(waveform) - samples) if training else (len(waveform) - samples) // 2
        waveform = waveform[start:start + samples]
    elif len(waveform) < samples:
        waveform = np.pad(waveform, (0, samples - len(waveform)), mode="wrap")
    if training and augment is not None and random.random() < 0.6:
        waveform = augment(waveform, sr=target_sr)
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 1e-6:
        waveform = waveform * (0.1 / rms)
    return waveform.astype(np.float32)


def audio_windows(waveform, rate=16000):
    """Four seconds, two-second hop, final window covering the end."""
    size, hop = rate * 4, rate * 2
    last = max(0, len(waveform) - size)
    starts = sorted(set([*range(0, last + 1, hop), last]))
    return [(start / rate, prepare_audio(waveform[start:start + size], rate))
            for start in starts]


def decode_audio(path, ffmpeg):
    result = subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-t", "31",
        "-f", "f32le", "pipe:1",
    ], capture_output=True, timeout=60, check=False)
    if result.returncode:
        raise ValueError("Audio decoding failed: " + result.stderr.decode(errors="replace")[-500:])
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not 1600 <= len(waveform) <= 30 * 16000 or not np.isfinite(waveform).all():
        raise ValueError("Expected finite audio lasting 0.1 to 30 seconds")
    return waveform
