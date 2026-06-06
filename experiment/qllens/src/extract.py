import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

IMAGE_PAD_TOKEN_ID = 151655  # <|image_pad|> in Qwen2-VL vocabulary
FIXED_RESOLUTION = 448


def get_patch_info(processor, image: Image.Image, image_idx: int, cfg) -> dict:
    """
    Phase 0a — processor only, no model weights.
    Determines patch_idx and pixel bbox, saves bbox overlay PNG.

    Returns dict with: patch_idx, bbox, grid, vision_start, num_vision.
    """
    processed = _center_crop_square(image).resize(
        (FIXED_RESOLUTION, FIXED_RESOLUTION), Image.LANCZOS
    )

    inputs = processor(
        images=[processed],
        text="<|image_pad|>Describe this image.",
        return_tensors="pt",
    )

    # Locate vision tokens
    input_ids = inputs["input_ids"][0]
    positions = (input_ids == IMAGE_PAD_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert len(positions) > 0, (
        f"No IMAGE_PAD tokens found (id={IMAGE_PAD_TOKEN_ID}). "
        "Check processor tokenization."
    )
    num_vision = len(positions)
    vision_start = int(positions[0])

    # Derive the MERGED token grid.
    # image_grid_thw holds the PRE-merge grid: at 448px it is [1, 32, 32].
    # Actual visual-token grid = (grid_h // merge_size, grid_w // merge_size) = (16, 16).
    thw = inputs["image_grid_thw"]   # shape [1, 3]
    _, gh, gw = thw[0].tolist()
    merge_size = processor.image_processor.merge_size   # = 2
    H = int(gh) // merge_size
    W = int(gw) // merge_size
    assert H * W == num_vision, (
        f"Merged grid {H}×{W}={H*W} != num_vision_tokens {num_vision}. "
        f"image_grid_thw={thw.tolist()}, merge_size={merge_size}. "
        "Processor resolution lock may have failed."
    )
    merged_patch_px = FIXED_RESOLUTION // W   # = 28 = patch_size(14) × merge_size(2)

    # Select from top-10 highest-variance patches (most visual content)
    img_arr = np.array(processed.convert("RGB"))
    variances = [
        float(img_arr[r * merged_patch_px:(r + 1) * merged_patch_px,
                       c * merged_patch_px:(c + 1) * merged_patch_px].var())
        for r in range(H) for c in range(W)
    ]
    top_indices = sorted(range(H * W), key=lambda i: variances[i], reverse=True)[:10]
    patch_idx = random.Random(cfg.seed + image_idx).choice(top_indices)

    # Bbox in pixel coordinates on the 448×448 processed image
    row = patch_idx // W
    col = patch_idx % W
    x1 = col * merged_patch_px
    y1 = row * merged_patch_px
    x2 = x1 + merged_patch_px
    y2 = y1 + merged_patch_px

    # Save bbox overlay (the visual gate for Phase 0)
    figures_dir = cfg.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    overlay = processed.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline="white", width=2)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    overlay_path = figures_dir / f"smoke_bbox_{image_idx}.png"
    overlay.save(overlay_path)
    print(
        f"    patch_idx={patch_idx} row={row} col={col} "
        f"bbox=({x1},{y1},{x2},{y2}) → {overlay_path.name}"
    )

    return {
        "patch_idx": patch_idx,
        "vision_start": vision_start,
        "bbox": [x1, y1, x2, y2],
        "grid": [H, W],
        "num_vision": num_vision,
        "merged_patch_px": merged_patch_px,
    }


def extract_hidden_states(
    model, processor, image: Image.Image, patch_info: dict, cfg, device
) -> dict:
    """
    Phase 0b — full forward pass.
    Returns {visual_layer: Tensor[hidden_dim]} for all cfg.visual_layers.

    Layer indexing (documented once):
      hidden_states[0] = embedding layer output  (visual_layer=0 in config)
      hidden_states[i] = transformer block i output  (visual_layers 1,2,4,...,27)
    This matches the reference index's layer numbering directly.
    """
    processed = _center_crop_square(image).resize(
        (FIXED_RESOLUTION, FIXED_RESOLUTION), Image.LANCZOS
    )

    inputs = processor(
        images=[processed],
        text="<|image_pad|>Describe this image.",
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states   # tuple len = 1 + num_blocks
    vision_start = patch_info["vision_start"]
    patch_idx = patch_info["patch_idx"]

    result = {}
    for layer in cfg.visual_layers:
        if layer >= len(hidden_states):
            print(f"  Warning: visual_layer={layer} out of range ({len(hidden_states)} total)")
            continue
        hs = hidden_states[layer]                        # [1, seq_len, hidden_dim]
        token_pos = vision_start + patch_idx
        result[layer] = hs[0, token_pos, :].float().cpu()

    return result


def _center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    s = min(w, h)
    return image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
