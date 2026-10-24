"""
VeriSense: Prepare Image Data
Walks through raw image directories (GenImage style or real/fake folders),
verifies image integrity (discarding truncated files), and indexes metadata.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def verify_image(file_path: Path) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Ensure it can also be fully loaded
        with Image.open(file_path) as img:
            img.load()
            return img.size[0] >= 32 and img.size[1] >= 32
    except Exception:
        return False


def scan_image_directory(input_dir: Path):
    records = []
    print(f"Scanning directory: {input_dir}")
    
    # Check if standard real / fake structure exists
    subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
    sub_names = {p.name.lower() for p in subdirs}

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if "real" in sub_names or "fake" in sub_names:
        for folder in subdirs:
            lbl_name = folder.name.lower()
            label = 0 if "real" in lbl_name else 1
            generator = "nature" if label == 0 else folder.name
            
            for file_path in folder.rglob("*"):
                if file_path.suffix.lower() in image_extensions:
                    records.append({
                        "file_path": str(file_path.resolve()),
                        "label": label,
                        "generator": generator,
                        "format": file_path.suffix.lower().replace(".", "")
                    })
    else:
        # GenImage or flat layout
        for file_path in input_dir.rglob("*"):
            if file_path.suffix.lower() in image_extensions:
                parts = [p.lower() for p in file_path.parts]
                is_fake = any(k in parts for k in ["fake", "ai", "synthesized", "generated"])
                label = 1 if is_fake else 0
                generator = file_path.parent.name if label == 1 else "authentic"
                
                records.append({
                    "file_path": str(file_path.resolve()),
                    "label": label,
                    "generator": generator,
                    "format": file_path.suffix.lower().replace(".", "")
                })

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate image dataset for VeriSense")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to raw image directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/images", help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Verify PIL decode integrity for all files")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = scan_image_directory(input_path)
    print(f"Found {len(records)} candidate images.")

    if args.verify:
        print("Verifying image integrity...")
        valid_records = []
        for r in tqdm(records):
            if verify_image(Path(r["file_path"])):
                valid_records.append(r)
        records = valid_records
        print(f"Verified {len(records)} clean images.")

    df = pd.DataFrame(records)
    csv_path = output_path / "all_images_manifest.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved image manifest to {csv_path}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
