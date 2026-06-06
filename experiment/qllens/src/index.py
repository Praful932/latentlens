from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Bank:
    # Kept fp16 on CPU — normalized on GPU at search time (avoids doubling RAM)
    embeddings: torch.Tensor   # [N, 3584] float16
    metadata: list             # [{token_str, token_id, caption, position}, ...]


def load_banks(layers: list, cfg) -> dict:
    """Load reference embedding banks from local .pt files. Returns {layer: Bank}."""
    banks = {}
    for L in layers:
        path = Path(cfg.index_dir) / f"layer_{L}" / "embeddings_cache.pt"
        print(f"  Loading layer {L} from {path.name}...", end=" ", flush=True)
        data = torch.load(path, map_location="cpu", weights_only=False)
        banks[L] = Bank(embeddings=data["embeddings"], metadata=data["metadata"])
        e = data["embeddings"]
        print(f"shape={list(e.shape)} dtype={e.dtype}")
    return banks
