#!/usr/bin/env python3
"""
Usage (from qllens/ directory):
  python -m src.run --phase 0a   # fetch images + bbox overlays (no model, runs anywhere)
  python -m src.run --phase 0b   # model + NN search + WordNet  (needs GPU for fp16 7B)
"""
import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src import data as data_mod
from src import models as models_mod
from src import index as index_mod
from src import extract as extract_mod
from src import nn_search
from src import wordnet_h2


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_0a(cfg):
    """
    Processor-only smoke test — runs on CPU/MPS/CUDA in seconds.
    Fetches 5 images, computes bbox for each, saves overlay PNGs + smoke_patches.json.
    Hard gate: verify results/figures/smoke_bbox_*.png before running phase 0b.
    """
    print("\n=== Phase 0a: Image fetch + bbox overlays ===\n")

    print("Fetching images from pixmo_cap validation split...")
    records = data_mod.get_images(cfg)
    data_mod.save_image_metadata(records, cfg.results_dir)

    print("\nLoading processor (no model weights)...")
    processor = models_mod.load_processor(cfg)

    print("\nComputing patch info + drawing bbox overlays...")
    patches = []
    for i, rec in enumerate(records):
        print(f"\n  Image {i + 1}/{len(records)} — val_idx={rec['idx']}")
        info = extract_mod.get_patch_info(processor, rec["image"], i, cfg)
        patches.append({
            "image_num": i,
            "idx": rec["idx"],
            "image_url": rec["image_url"],
            "caption": rec["caption"][:120],
            "patch_idx": info["patch_idx"],
            "bbox": info["bbox"],
            "grid": info["grid"],
            "vision_start": info["vision_start"],
            "num_vision": info["num_vision"],
        })

    patches_path = cfg.results_dir / "smoke_patches.json"
    patches_path.parent.mkdir(parents=True, exist_ok=True)
    with open(patches_path, "w") as f:
        json.dump(patches, f, indent=2)

    print(f"\n✓ Phase 0a done.")
    print(f"  Patches saved → {patches_path}")
    print(f"  Bbox overlays → {cfg.results_dir / 'figures'}/smoke_bbox_*.png")
    print(f"\n  *** Open the overlay PNGs and verify the red box lands on a")
    print(f"  *** recognizable region before running phase 0b.")
    print(f"\n  Next: python -m src.run --phase 0b")


def phase_0b(cfg):
    """
    Full smoke test — needs GPU (Kaggle T4) for fp16 7B model.
    Reads smoke_patches.json from phase 0a; extracts hidden states, does NN search,
    WordNet scoring. Outputs smoke_fp16.jsonl + smoke_depth_scores.csv.
    """
    print("\n=== Phase 0b: Model + NN search + WordNet ===\n")

    patches_path = cfg.results_dir / "smoke_patches.json"
    if not patches_path.exists():
        print(f"ERROR: {patches_path} not found. Run --phase 0a first.")
        sys.exit(1)
    with open(patches_path) as f:
        patches = json.load(f)

    print("Re-fetching images...")
    metadata = data_mod.load_image_metadata(cfg.results_dir)
    records = data_mod.reload_images(metadata)

    print("\nLoading model (fp16)...")
    model, device = models_mod.load_model("fp16", cfg)
    processor = models_mod.load_processor(cfg)
    print(f"  Device: {device}")

    print(f"\nLoading reference banks {cfg.reference_layers_smoke}...")
    banks = index_mod.load_banks(cfg.reference_layers_smoke, cfg)

    _ensure_nlp_resources()

    neighbors_dir = cfg.results_dir / "neighbors"
    neighbors_dir.mkdir(parents=True, exist_ok=True)
    h2_dir = cfg.results_dir / "h2"
    h2_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = neighbors_dir / "smoke_fp16.jsonl"
    csv_path = h2_dir / "smoke_depth_scores.csv"

    csv_rows = []
    sanity = []

    with open(jsonl_path, "w") as jf:
        for i, (rec, patch_info) in enumerate(zip(records, patches)):
            print(f"\n  Image {i + 1}/{len(records)} — val_idx={patch_info['idx']}")

            hs_by_layer = extract_mod.extract_hidden_states(
                model, processor, rec["image"], patch_info, cfg, device
            )

            for layer in cfg.visual_layers:
                if layer not in hs_by_layer:
                    continue
                neighbors = nn_search.search(hs_by_layer[layer], banks, cfg.top_k, device)
                depth_result = wordnet_h2.score_neighbors(neighbors)

                jf.write(json.dumps({
                    "condition": "fp16",
                    "image_num": i,
                    "val_idx": patch_info["idx"],
                    "image_url": patch_info["image_url"],
                    "patch_idx": patch_info["patch_idx"],
                    "bbox": patch_info["bbox"],
                    "grid": patch_info["grid"],
                    "visual_layer": layer,
                    "neighbors": [
                        {
                            "token_str": n.token_str,
                            "similarity": round(n.similarity, 4),
                            "contextual_layer": n.contextual_layer,
                            "caption": n.caption,
                            "position": n.position,
                        }
                        for n in neighbors
                    ],
                }) + "\n")

                csv_rows.append({
                    "image_num": i,
                    "val_idx": patch_info["idx"],
                    "visual_layer": layer,
                    "n_valid": depth_result["n_valid"],
                    "mean_depth": depth_result["mean_depth"],
                    "depths": str(depth_result["depths"]),
                })

                if neighbors:
                    top = neighbors[0]
                    sanity.append((i, layer, top.token_str.strip(), top.contextual_layer, round(top.similarity, 3)))

    with open(csv_path, "w", newline="") as cf:
        if csv_rows:
            writer = csv.DictWriter(cf, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    _print_sanity_summary(sanity)

    n_rows = sum(1 for _ in open(jsonl_path))
    n_valid_vals = [r["n_valid"] for r in csv_rows]
    print(f"\n✓ Phase 0b done.")
    print(f"  Neighbors:    {jsonl_path}  ({n_rows} rows, expected {len(records) * len(cfg.visual_layers)})")
    print(f"  Depth scores: {csv_path}")
    if n_valid_vals:
        zeros = sum(1 for v in n_valid_vals if v == 0)
        print(f"  WordNet n_valid: mean={sum(n_valid_vals)/len(n_valid_vals):.1f}, "
              f"zeros={zeros}/{len(n_valid_vals)}")
        if zeros == len(n_valid_vals):
            print("  WARNING: n_valid=0 for ALL rows — check word reconstruction in wordnet_h2.py")


def _print_sanity_summary(sanity):
    print("\n\n--- Sanity: top-1 neighbor per (image, visual_layer) ---")
    print(f"{'img':>4} {'layer':>6}  {'neighbor':<22} {'ctx_L':>6} {'sim':>7}")
    print("-" * 54)
    for img_i, layer, word, ctx_l, sim in sanity:
        print(f"{img_i:>4} {layer:>6}  {word:<22} {ctx_l:>6} {sim:>7.3f}")


def _ensure_nlp_resources():
    print("\nChecking NLP resources...")
    import nltk
    try:
        wn = __import__("nltk.corpus", fromlist=["wordnet"]).wordnet
        wn.synsets("dog")
    except Exception:
        print("  Downloading NLTK wordnet + omw-1.4...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("  Downloading spaCy en_core_web_sm...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

    print("  NLP resources OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["0a", "0b"], required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: qllens/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg.seed)

    if args.phase == "0a":
        phase_0a(cfg)
    elif args.phase == "0b":
        phase_0b(cfg)


if __name__ == "__main__":
    main()
