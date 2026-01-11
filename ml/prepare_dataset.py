"""Utility to consolidate raw Indian-food datasets into train/val/test splits.

Usage example:

    python ml/prepare_dataset.py \
        --manifest datasets_manifest.json \
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

The manifest JSON expects:
{
  "sources": [
    {"name": "kaggle_indian_food", "path": "data/raw/kaggle_indian_food"},
    {"name": "open_images_curries", "path": "data/raw/open_images_curries"}
  ]
}

Each ``path`` should point to a directory that contains class folders
(``<class_name>/<image files>``). Images are copied into
``data/indian_food/{train,val,test}/<class_name>/``.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "indian_food"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    data = json.loads(manifest_path.read_text())
    sources = data.get("sources", [])
    if not sources:
        raise ValueError("Manifest must contain a non-empty 'sources' array.")
    return sources


def collect_images(source_dir: Path) -> Dict[str, List[Path]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    class_map: Dict[str, List[Path]] = {}
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower().replace(" ", "_")
        files = [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if files:
            class_map.setdefault(class_name, []).extend(files)
    return class_map


def split_files(
    files: Sequence[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Path]]:
    if not 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01:
        raise ValueError("Split ratios must sum to 1.0")

    files = list(files)
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }


def copy_images(split_map: Dict[str, List[Path]], class_name: str) -> None:
    for split, paths in split_map.items():
        target_dir = OUTPUT_DIR / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            target_path = target_dir / path.name
            if target_path.exists():
                target_path = target_dir / f"{path.stem}_{abs(hash(path))}{path.suffix}"
            shutil.copy2(path, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the dataset manifest JSON.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_manifest(args.manifest)
    aggregated: Dict[str, List[Path]] = {}

    for source in sources:
        source_path = Path(source.get("path", ""))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        print(f"Scanning {source['name']} at {source_path} ...")
        source_map = collect_images(source_path)
        for class_name, paths in source_map.items():
            aggregated.setdefault(class_name, []).extend(paths)

    print(f"Found {len(aggregated)} classes across {len(sources)} sources.")

    for class_name, files in aggregated.items():
        if len(files) < 10:
            print(f"Skipping '{class_name}' (not enough samples: {len(files)}).")
            continue
        split_map = split_files(
            files,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        copy_images(split_map, class_name)
        print(
            f"{class_name}: train={len(split_map['train'])}, "
            f"val={len(split_map['val'])}, test={len(split_map['test'])}"
        )

    print(f"Finished preparing dataset under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
