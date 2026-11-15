"""Configure and warm-start the existing temporal trainer on LAV-DF NPZ caches."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video.train import train_video_model


def adapt(cache_dir, warm_start, output_dir):
    checkpoint = torch.load(warm_start, map_location="cpu", weights_only=True)
    config = copy.deepcopy(checkpoint["config"])
    if config["temporal_model"]["max_frames"] != 16:
        raise ValueError("This experiment uses the existing 16-frame cache and temporal model")
    config["data"].update(train_split_csv=str(cache_dir / "train.csv"),
                          val_split_csv=str(cache_dir / "val.csv"), num_workers=0,
                          batch_size=16, val_batch_size=32, preload_embeddings=True,
                          cache_provenance=json.loads((cache_dir / "provenance.json").read_text()))
    config["data"].pop("test_split_csv", None)
    config["training"].update(epochs=15, lr=.0001, early_stopping_patience=4,
                              gradient_accumulation_steps=1, checkpoint_dir=str(output_dir),
                              checkpoint_name="video_lavdf_best.pt", frame_supervision="annotated")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return train_video_model(path, warm_start=warm_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    adapt(args.cache_dir, args.warm_start, args.output_dir)
