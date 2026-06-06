from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Neighbor:
    token_str: str
    similarity: float
    contextual_layer: int
    caption: str
    position: int


def search(query: torch.Tensor, banks: dict, top_k: int, device: torch.device) -> list:
    """
    Global top-k nearest neighbors across all loaded reference layers.

    For each bank: moves fp16 embeddings to GPU, normalizes, does cosine search,
    then frees GPU memory. No fp32 CPU copy is ever stored.

    query: Tensor[hidden_dim], any dtype, any device.
    Returns list[Neighbor] sorted by similarity descending, length = top_k.
    """
    query_norm = F.normalize(query.float(), dim=-1).to(device)

    candidates = []
    for layer, bank in banks.items():
        emb = bank.embeddings.to(device).float()
        emb_norm = F.normalize(emb, dim=-1)
        sims = query_norm @ emb_norm.T   # [N]

        k = min(top_k, sims.shape[0])
        vals, idxs = torch.topk(sims, k=k)

        for sim, idx in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            meta = bank.metadata[idx]
            candidates.append(Neighbor(
                token_str=meta["token_str"],
                similarity=sim,
                contextual_layer=layer,
                caption=meta["caption"],
                position=meta["position"],
            ))

        del emb, emb_norm, sims
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidates.sort(key=lambda n: n.similarity, reverse=True)
    return candidates[:top_k]
