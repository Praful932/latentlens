#!/usr/bin/env python3
"""
Prepare PixMo-Cap train/val split matching the exact parameters used in the
LatentLens paper codebase (molmo/data/pixmo_datasets.py).

The HuggingFace dataset allenai/pixmo-cap only has a 'train' split. The paper
carves out a local validation set using:
    dataset.train_test_split(test_size=2048, seed=96817)

This script replicates that split and saves both halves to disk so the experiment
can load them without depending on the molmo codebase.

Output layout:
    data/pixmo_cap/
        train/          <- HF DatasetDict "train" shard
        validation/     <- HF DatasetDict "validation" shard (2048 examples)
        split_info.json <- record of split parameters for reproducibility

Usage:
    python prepare_pixmo_cap.py [--output-dir OUTPUT_DIR] [--hf-cache-dir DIR]

Requirements:
    pip install datasets pillow
"""

import argparse
import json
import sys
from pathlib import Path

import datasets


# Exact parameters from molmo/data/pixmo_datasets.py save_local_dataset()
SPLIT_TEST_SIZE = 2048
SPLIT_SEED = 96817
HF_DATASET = "allenai/pixmo-cap"
HF_SPLIT = "train"


def prepare(output_dir: Path, hf_cache_dir: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "pixmo_cap"

    if save_path.exists():
        print(f"Split already exists at {save_path}. Delete it to re-run.")
        _print_stats(save_path)
        return

    print(f"Loading {HF_DATASET} ({HF_SPLIT} split) from HuggingFace...")
    ds = datasets.load_dataset(
        HF_DATASET,
        split=HF_SPLIT,
        cache_dir=hf_cache_dir,
    )
    print(f"  Loaded {len(ds):,} examples")
    print(f"  Columns: {ds.column_names}")

    print(f"\nApplying train_test_split(test_size={SPLIT_TEST_SIZE}, seed={SPLIT_SEED})...")
    split = ds.train_test_split(test_size=SPLIT_TEST_SIZE, seed=SPLIT_SEED)
    dataset_dict = datasets.DatasetDict(
        train=split["train"],
        validation=split["test"],
    )
    print(f"  train:      {len(dataset_dict['train']):,} examples")
    print(f"  validation: {len(dataset_dict['validation']):,} examples")

    print(f"\nSaving to {save_path} ...")
    dataset_dict.save_to_disk(str(save_path))

    split_info = {
        "hf_dataset": HF_DATASET,
        "hf_split": HF_SPLIT,
        "test_size": SPLIT_TEST_SIZE,
        "seed": SPLIT_SEED,
        "train_size": len(dataset_dict["train"]),
        "validation_size": len(dataset_dict["validation"]),
        "columns": ds.column_names,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\nDone.")
    _print_stats(save_path)


def _print_stats(save_path: Path) -> None:
    dd = datasets.load_from_disk(str(save_path))
    print(f"\nLoaded from disk:")
    for split_name, split_ds in dd.items():
        print(f"  {split_name}: {len(split_ds):,} examples  |  columns: {split_ds.column_names}")
    val = dd["validation"]
    print(f"\nFirst validation example keys: {list(val[0].keys())}")
    if "image_url" in val.column_names:
        print(f"  image_url[0]: {val[0]['image_url']}")
    if "caption" in val.column_names:
        snippet = val[0]["caption"][:120].replace("\n", " ")
        print(f"  caption[0]:   {snippet}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PixMo-Cap train/val split")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to save the split (default: same directory as this script)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    prepare(Path(args.output_dir), args.hf_cache_dir)
