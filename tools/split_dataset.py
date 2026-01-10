"""Utility script to split raw class folders into train/val/test directories.

Usage
-----
1. Drop all images for each dish into ``data/raw/<class_name>/``.
2. Adjust ``SPLIT_RATIOS`` if needed.
3. Run ``python tools/split_dataset.py`` from the project root.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

RANDOM_SEED = 1337
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
RAW_ROOT = Path("data/raw")
OUTPUT_ROOT = Path("data/indian_food")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(class_dir: Path) -> list[Path]:
    """Return a list of image files under ``class_dir`` filtered by extension."""
    files = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(files)


def _assign_splits(file_count: int) -> tuple[int, int]:
    """Compute cut points for train/val/test according to SPLIT_RATIOS."""
    train_cut = int(file_count * SPLIT_RATIOS["train"])
    val_cut = train_cut + int(file_count * SPLIT_RATIOS["val"])
    return train_cut, val_cut


def _copy_files(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, destination / src.name)


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw data directory '{RAW_ROOT}' not found.")

    random.seed(RANDOM_SEED)

    for class_dir in RAW_ROOT.iterdir():
        if not class_dir.is_dir():
            continue

        images = _collect_images(class_dir)
        if not images:
            print(f"Skipping '{class_dir.name}' (no images found).")
            continue

        random.shuffle(images)
        train_cut, val_cut = _assign_splits(len(images))

        split_map = {
            "train": images[:train_cut],
            "val": images[train_cut:val_cut],
            "test": images[val_cut:],
        }

        for split, files in split_map.items():
            target_dir = OUTPUT_ROOT / split / class_dir.name
            _copy_files(files, target_dir)
            print(f"{class_dir.name}: copied {len(files)} files to {target_dir}")

    print("Split complete. Ready to run ml/food_classifier_training.py")


if __name__ == "__main__":
    main()
