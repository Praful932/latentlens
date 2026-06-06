from typing import Optional

import spacy
from nltk.corpus import wordnet as wn

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _reconstruct_word(token_str: str, caption: str, position: int) -> str:
    """
    Expand a BPE subword token to the full surface word using its caption context.
    'position' is the character offset of the token in the caption.
    """
    token_clean = token_str.strip().lower()
    if not token_clean:
        return ""

    caption_lower = caption.lower()

    # Search near the recorded position first, then fall back to first occurrence
    idx = caption_lower.find(token_clean, max(0, position - 2))
    if idx == -1:
        idx = caption_lower.find(token_clean)
    if idx == -1:
        return token_clean  # use token as-is

    # Expand to word boundaries (alphabetic characters only)
    start = idx
    while start > 0 and caption[start - 1].isalpha():
        start -= 1
    end = idx + len(token_clean)
    while end < len(caption) and caption[end].isalpha():
        end += 1

    return caption[start:end].lower()


def _min_depth(word: str) -> Optional[int]:
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return None
    return min(s.min_depth() for s in synsets)


def score_neighbors(neighbors: list) -> dict:
    """
    Compute WordNet min_depth for top-k neighbors.
    Only nouns (NOUN/PROPN via spaCy) are scored; others get depth=None.

    Returns:
        depths:     list[Optional[int]] aligned to neighbors
        n_valid:    number of neighbors that yielded a depth
        mean_depth: mean of valid depths, or None if none valid
    """
    nlp = _get_nlp()
    depths = []

    for n in neighbors:
        word = _reconstruct_word(n.token_str, n.caption, n.position)
        if not word:
            depths.append(None)
            continue

        doc = nlp(word)
        is_noun = any(tok.pos_ in ("NOUN", "PROPN") for tok in doc)
        if not is_noun:
            depths.append(None)
            continue

        depths.append(_min_depth(word))

    n_valid = sum(1 for d in depths if d is not None)
    mean_depth = (
        sum(d for d in depths if d is not None) / n_valid if n_valid > 0 else None
    )
    return {"depths": depths, "n_valid": n_valid, "mean_depth": mean_depth}
