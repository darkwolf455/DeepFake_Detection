"""
VeriSense: Prepare Audio Data
Extracts, resamples to 16kHz mono, and normalizes audio waveforms from raw audio or video files.
Saves standardized audio chunks and an indexed manifest.
"""

import argparse
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def process_audio_file(
    input_path: Path,
    output_dir: Path,
    target_sr: int = 16000,
    target_duration: float = 4.0,
):
    try:
        # Load audio and resample to 16kHz mono
        waveform, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        if len(waveform) == 0:
            return None

        # Normalize RMS amplitude to -20 dBFS
        rms = np.sqrt(np.mean(waveform**2))
        if rms > 1e-6:
            waveform = waveform * (0.1 / rms)
        waveform = np.clip(waveform, -1.0, 1.0)

        # Segment or pad to target_duration
        target_samples = int(target_sr * target_duration)
        if len(waveform) < target_samples:
            pad_width = target_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), mode="wrap")
        else:
            waveform = waveform[:target_samples]

        out_name = f"{input_path.stem}_16k.wav"
        out_path = output_dir / out_name
        sf.write(str(out_path), waveform, target_sr, subtype="PCM_16")

        return {
            "audio_id": input_path.stem,
            "raw_path": str(input_path.resolve()),
            "processed_path": str(out_path.resolve()),
            "sample_rate": target_sr,
            "duration": target_duration,
            "num_samples": len(waveform),
        }
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Normalize and chunk audio for VeriSense")
    parser.add_argument("--input-dir", type=str, required=True, help="Input raw audio directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/audio", help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
    audio_files = [p for p in input_path.rglob("*") if p.suffix.lower() in audio_extensions]
    print(f"Discovered {len(audio_files)} audio files.")

    records = []
    for f in tqdm(audio_files):
        rec = process_audio_file(f, output_path, args.sample_rate, args.duration)
        if rec is not None:
            # Determine label and speaker from path or filename
            parts = [p.lower() for p in f.parts]
            is_spoof = any(k in parts or k in f.stem.lower() for k in ["spoof", "fake", "cloned", "tts", "vc"])
            rec["label"] = 1 if is_spoof else 0
            
            # Speaker ID parsing (e.g. ASVspoof speaker ID is typically first token)
            tokens = f.stem.split("_")
            rec["speaker_id"] = tokens[0] if len(tokens) > 1 else "spk_unknown"
            records.append(rec)

    df = pd.DataFrame(records)
    manifest_file = output_path / "audio_manifest.csv"
    df.to_csv(manifest_file, index=False)
    print(f"Saved {len(df)} records to {manifest_file}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
