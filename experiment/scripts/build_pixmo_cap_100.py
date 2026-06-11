#!/usr/bin/env python3
"""
Build a 100-image PixMo-Cap validation subset with downloaded RGB images embedded
in the dataset, then push to the Hugging Face Hub.

Source: local train/val split from prepare_pixmo_cap.py (paper-exact split).
Selection: scan validation rows in order (val_idx 0, 1, 2, …) and keep the first
N rows whose image_url downloads successfully — same logic as 4_fp_baseline.ipynb.

Output columns (original + image + val_idx for reproducibility):
    val_idx, image_url, caption, transcripts, image

Usage:
    # Build locally, print stats, skip Hub push
    python build_pixmo_cap_100.py --no-push

    # Build and push (requires HF login or HF_TOKEN)
    python build_pixmo_cap_100.py \
        --repo-id McGill-NLP/latentlens-pixmo-cap-val100

    python build_pixmo_cap_100.py \
        --pixmo-dir /workspace/latentlens/experiment/data/pixmo_cap \
        --output-dir /workspace/latentlens/experiment/data/pixmo_cap_val100 \
        --repo-id org/my-dataset \
        --num-images 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import datasets
import requests
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PIXMO_DIR = SCRIPT_DIR.parent / "data" / "pixmo_cap"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "pixmo_cap_val100"
DEFAULT_REPO_ID = "McGill-NLP/latentlens-pixmo-cap-val100"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; latentlens-research/1.0)"}
SOURCE_SPLIT = "validation"
TARGET_SPLIT = "validation"


def fetch_image(url: str, timeout: float, retries: int, retry_backoff: float) -> PILImage.Image:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout, headers=_HEADERS)
            response.raise_for_status()
            return PILImage.open(BytesIO(response.content)).convert("RGB")
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(retry_backoff * (2**attempt))
    assert last_err is not None
    raise last_err


def build_subset(
    pixmo_dir: Path,
    num_images: int,
    *,
    timeout: float,
    retries: int,
    retry_backoff: float,
    request_delay: float,
    start_idx: int,
) -> tuple[Dataset, dict]:
    dd = datasets.load_from_disk(str(pixmo_dir))
    if SOURCE_SPLIT not in dd:
        raise KeyError(
            f"Expected '{SOURCE_SPLIT}' split in {pixmo_dir}; found {list(dd.keys())}"
        )
    source = dd[SOURCE_SPLIT]

    rows: list[dict] = []
    skipped: list[dict] = []
    idx = start_idx

    pbar = tqdm(total=num_images, desc="Downloading images")
    while len(rows) < num_images and idx < len(source):
        row = source[idx]
        url = row["image_url"]
        try:
            if request_delay > 0 and (rows or skipped):
                time.sleep(request_delay)
            image = fetch_image(url, timeout=timeout, retries=retries, retry_backoff=retry_backoff)
            rows.append(
                {
                    "val_idx": idx,
                    "image_url": url,
                    "caption": row["caption"],
                    "transcripts": row["transcripts"],
                    "image": image,
                }
            )
            pbar.update(1)
            pbar.set_postfix(val_idx=idx, ok=len(rows))
        except Exception as exc:
            skipped.append({"val_idx": idx, "image_url": url, "error": str(exc)})
            tqdm.write(f"  Skipped val_idx={idx}: {exc}")
        idx += 1
    pbar.close()

    if len(rows) < num_images:
        raise RuntimeError(
            f"Only downloaded {len(rows)}/{num_images} images before exhausting "
            f"validation split (scanned through val_idx={idx - 1})."
        )

    features = Features(
        {
            "val_idx": Value("int32"),
            "image_url": Value("string"),
            "caption": Value("string"),
            "transcripts": Sequence(Value("string")),
            "image": Image(),
        }
    )
    ds = Dataset.from_list(rows, features=features)

    meta = {
        "source_dataset": str(pixmo_dir.resolve()),
        "source_split": SOURCE_SPLIT,
        "target_split": TARGET_SPLIT,
        "num_images_requested": num_images,
        "num_images_downloaded": len(rows),
        "num_skipped": len(skipped),
        "start_idx": start_idx,
        "end_val_idx": rows[-1]["val_idx"],
        "val_indices": [r["val_idx"] for r in rows],
        "skipped": skipped,
        "columns": ds.column_names,
    }
    return ds, meta


def print_stats(ds: Dataset, meta: dict) -> None:
    print("\n" + "─" * 60)
    print("Dataset stats")
    print("─" * 60)
    print(f"  Examples:     {len(ds):,}")
    print(f"  Columns:      {ds.column_names}")
    print(f"  val_idx range: {meta['val_indices'][0]} … {meta['val_indices'][-1]}")
    print(f"  Skipped URLs: {meta['num_skipped']}")

    widths: list[int] = []
    heights: list[int] = []
    caption_lens: list[int] = []
    transcript_counts: list[int] = []

    for ex in ds:
        w, h = ex["image"].size
        widths.append(w)
        heights.append(h)
        caption_lens.append(len(ex["caption"]))
        transcript_counts.append(len(ex["transcripts"]))

    print(f"  Image width:  min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.0f}")
    print(f"  Image height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.0f}")
    print(f"  Caption len:  min={min(caption_lens)}, max={max(caption_lens)}, mean={sum(caption_lens)/len(caption_lens):.0f}")
    print(
        f"  Transcripts:  min={min(transcript_counts)}, max={max(transcript_counts)}, "
        f"mean={sum(transcript_counts)/len(transcript_counts):.1f}"
    )

    print("\nFirst example:")
    ex0 = ds[0]
    print(f"  val_idx:    {ex0['val_idx']}")
    print(f"  image_url:  {ex0['image_url'][:90]}...")
    print(f"  image size: {ex0['image'].size}")
    snippet = ex0["caption"][:120].replace("\n", " ")
    print(f"  caption:    {snippet}...")
    print(f"  transcripts: {len(ex0['transcripts'])} item(s)")

    if meta["skipped"]:
        print(f"\nSkipped examples ({len(meta['skipped'])}):")
        for item in meta["skipped"][:5]:
            print(f"  val_idx={item['val_idx']}: {item['error']}")
        if len(meta["skipped"]) > 5:
            print(f"  … and {len(meta['skipped']) - 5} more")


def save_local(output_dir: Path, dataset_dict: DatasetDict, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    with open(output_dir / "subset_info.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved locally to {output_dir.resolve()}")


def push_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    *,
    token: str | None,
    private: bool,
) -> None:
    print(f"\nPushing to Hugging Face Hub: {repo_id}")
    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message="Add 100-image PixMo-Cap validation subset with embedded images",
    )
    print(f"  https://huggingface.co/datasets/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 100-image PixMo-Cap subset with embedded images and push to HF Hub.",
    )
    parser.add_argument(
        "--pixmo-dir",
        type=Path,
        default=DEFAULT_PIXMO_DIR,
        help=f"Local pixmo_cap DatasetDict directory (default: {DEFAULT_PIXMO_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save the built dataset locally (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of successfully downloaded images to include (default: 100)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Validation index to start scanning from (default: 0)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo to push to (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build and save locally only; do not push to the Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the Hub repo as private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (default: HF_TOKEN env or cached huggingface-cli login)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout per image download in seconds (default: 15)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per failed download (default: 2)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Base backoff in seconds between retries (default: 1.0)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.15,
        help="Delay in seconds between download attempts (default: 0.15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pixmo_dir.is_dir():
        print(f"PixMo directory not found: {args.pixmo_dir}", file=sys.stderr)
        sys.exit(1)
    if args.num_images <= 0:
        print("--num-images must be positive", file=sys.stderr)
        sys.exit(1)

    print(f"Source:      {args.pixmo_dir.resolve()} [{SOURCE_SPLIT}]")
    print(f"Target:      {args.num_images} images (scan from val_idx={args.start_idx})")

    ds, meta = build_subset(
        args.pixmo_dir,
        args.num_images,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        request_delay=args.request_delay,
        start_idx=args.start_idx,
    )
    dataset_dict = DatasetDict({TARGET_SPLIT: ds})

    print_stats(ds, meta)
    save_local(args.output_dir, dataset_dict, meta)

    if args.no_push:
        print("\nSkipping Hub push (--no-push).")
        return

    push_to_hub(
        dataset_dict,
        args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
