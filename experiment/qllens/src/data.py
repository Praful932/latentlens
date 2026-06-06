import json
from io import BytesIO
from pathlib import Path

import datasets
import requests
from PIL import Image


_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def get_images(cfg) -> list[dict]:
    """
    Walk pixmo_cap validation split indices 0,1,2,… in order (matching the paper's
    sequential selection) until n_smoke images are successfully fetched.
    Skips dead URLs rather than failing hard.
    """
    ds = datasets.load_from_disk(str(cfg.pixmo_cap_dir))["validation"]
    n = cfg.test_images_n_smoke
    results = []
    idx = 0

    while len(results) < n and idx < len(ds):
        row = ds[idx]
        try:
            image = _fetch_image(row["image_url"])
            results.append({
                "idx": idx,
                "image_url": row["image_url"],
                "image": image,
                "caption": row["caption"],
            })
            print(f"  [{len(results)}/{n}] val_idx={idx} — {row['image_url'][:70]}")
        except Exception as e:
            print(f"  Skipped val_idx={idx}: {e}")
        idx += 1

    if len(results) < n:
        raise RuntimeError(f"Only found {len(results)}/{n} usable images in pixmo_cap val")

    return results


def _fetch_image(url: str) -> Image.Image:
    r = requests.get(url, stream=True, timeout=10, headers=_HEADERS)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def save_image_metadata(records: list[dict], results_dir: Path):
    out = results_dir / "smoke_images.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"idx": r["idx"], "image_url": r["image_url"], "caption": r["caption"][:120]}
        for r in records
    ]
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {out}")


def load_image_metadata(results_dir: Path) -> list[dict]:
    with open(results_dir / "smoke_images.json") as f:
        return json.load(f)


def reload_images(metadata: list[dict]) -> list[dict]:
    """Re-fetch images for phase 0b using URLs saved in smoke_images.json."""
    results = []
    for m in metadata:
        try:
            image = _fetch_image(m["image_url"])
            results.append({**m, "image": image})
            print(f"  Re-fetched val_idx={m['idx']}")
        except Exception as e:
            raise RuntimeError(f"Failed to re-fetch val_idx={m['idx']}: {e}") from e
    return results
