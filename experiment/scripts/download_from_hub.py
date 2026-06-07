#!/usr/bin/env python3
"""
Download a Hugging Face Hub repository into experiment/data/.

Uses snapshot_download with resume support. Skips files that are already
complete locally.

Usage:
    python download_index.py McGill-NLP/latentlens-qwen2vl-embeddings 
    python download_index.py --repo-id org/my-dataset --repo-type dataset
    python download_index.py org/my-model --output-dir ../data/my-model
    python download_index.py org/my-model --include "layer_*/embeddings_cache.pt"
    
    # embeddings
    python /workspace/latentlens/experiment/scripts/download_from_hub.py McGill-NLP/latentlens-qwen2vl-embeddings --output-dir /workspace/latentlens/experiment/data/latentlens-qwen2vl-embeddings
    # model
    python /workspace/latentlens/experiment/scripts/download_from_hub.py Qwen/Qwen2-VL-7B-Instruct --output-dir /workspace/latentlens/experiment/data/Qwen2-VL-7B-Instruct
    # index
    python /workspace/latentlens/experiment/scripts/prepare_pixmo_cap.py --output-dir /workspace/latentlens/experiment/data/
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"


def repo_basename(repo_id: str) -> str:
    return repo_id.strip().split("/")[-1]


def default_output_dir(repo_id: str) -> Path:
    return DEFAULT_DATA_DIR / repo_basename(repo_id)


def list_top_level_files(local_dir: Path, max_entries: int = 20) -> list[str]:
    if not local_dir.is_dir():
        return []
    entries = sorted(local_dir.rglob("*"))
    files = [str(p.relative_to(local_dir)) for p in entries if p.is_file()]
    return files[:max_entries]


def download_repo(
    repo_id: str,
    output_dir: Path,
    *,
    repo_type: str,
    revision: str | None,
    include: list[str] | None,
    exclude: list[str] | None,
    token: str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repo   : {repo_id} ({repo_type})")
    if revision:
        print(f"Revision: {revision}")
    print(f"Target : {output_dir.resolve()}")
    if include:
        print(f"Include: {include}")
    if exclude:
        print(f"Exclude: {exclude}")
    print()

    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=include or None,
        ignore_patterns=exclude or None,
        token=token,
    )
    return Path(path)


def print_summary(output_dir: Path, repo_id: str, repo_type: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"Summary — {output_dir.resolve()}")

    total_bytes = 0
    file_count = 0
    for p in output_dir.rglob("*"):
        if p.is_file():
            file_count += 1
            total_bytes += p.stat().st_size

    if file_count == 0:
        print("  No files found under output directory.")
        return

    print(f"  Files: {file_count}  Total: {total_bytes / 1e9:.2f} GB")

    try:
        api = HfApi()
        siblings = api.repo_info(repo_id, repo_type=repo_type).siblings or []
        if siblings:
            print(f"  Remote file count (repo metadata): {len(siblings)}")
    except Exception as exc:
        print(f"  (could not fetch remote metadata: {exc})")

    sample = list_top_level_files(output_dir)
    if sample:
        print("  Sample paths:")
        for rel in sample:
            p = output_dir / rel
            print(f"    {rel}  ({p.stat().st_size / 1e9:.3f} GB)")
        remaining = file_count - len(sample)
        if remaining > 0:
            print(f"    ... and {remaining} more file(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face Hub repo into experiment/data/",
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        help="Hub repo id, e.g. McGill-NLP/latentlens-qwen2vl-embeddings",
    )
    parser.add_argument(
        "--repo-id",
        dest="repo_id_flag",
        metavar="ID",
        help="Same as positional repo_id (use one or the other)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: experiment/data/<repo-basename>)",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Hub repository type (default: model)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit hash",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="GLOB",
        help="Only download paths matching this glob (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="GLOB",
        help="Skip paths matching this glob (repeatable)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (default: HF_TOKEN env or cached login)",
    )
    args = parser.parse_args()

    repo_id = args.repo_id_flag or args.repo_id
    if not repo_id:
        parser.error("repo_id is required (positional or --repo-id)")

    output_dir = args.output_dir or default_output_dir(repo_id)

    try:
        download_repo(
            repo_id,
            output_dir,
            repo_type=args.repo_type,
            revision=args.revision,
            include=args.include,
            exclude=args.exclude,
            token=args.token,
        )
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        sys.exit(1)

    print_summary(output_dir, repo_id, args.repo_type)


if __name__ == "__main__":
    main()
