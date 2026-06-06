# Quantization × LatentLens — Experiment Spec

A build spec for an experiment testing how LLM quantization affects the interpretability
of visual tokens in a VLM, using the LatentLens methodology (Krojer et al., 2026).

This document is the source of truth for implementation. Where it says **MUST**, do not
deviate without flagging. Where it says **default** or **may**, you (the implementer) can
choose a reasonable approach and note it.

---

## 0. Objective & hypotheses

We hold a **fixed full-precision (FP) contextual text index** as a stable semantic yardstick,
then extract visual-token representations from the **same VLM** under FP16, int8, and 4-bit
(NF4) quantization, and measure how interpretability changes.

- **H1 — interpretability degrades in later layers after quantization.**
  Metric: **% of interpretable visual tokens per layer** (paper's GPT judge).
  Prediction: int8/NF4 curves fall below FP, with the gap widening in later layers.

- **H2 — retrieved nearest-neighbor words shift from hyponyms toward hypernyms after quantization.**
  Metric: **WordNet `min_depth` of the top-5 neighbor words** (shallower = more hypernym-like).
  Prediction: mean `min_depth` decreases under quantization, more so in later layers.

We are testing **relative** FP-vs-quantized deltas within one model. Absolute numbers are
**not** meant to match the paper's (different judge config, our own run).

---

## 1. Experimental arms

There are **three conditions** (arms), differing only in how the LLM backbone is loaded. Everything else — images, patches, reference index, NN search, judge prompt, WordNet scoring — is identical across arms.

| Arm | Label | How loaded | Approx. VRAM | Notes |
|---|---|---|---|---|
| **A — Full precision** | `fp16` | `torch_dtype=torch.float16, device_map="auto"` | ~16.6 GB (needs both T4s) | Control / baseline. The reference index was built with this same model at FP16, so the embedding spaces are aligned by construction. |
| **B — 8-bit integer** | `int8` | `BitsAndBytesConfig(load_in_8bit=True)` | ~8 GB (single T4) | bitsandbytes LLM.int8(). Absorbable outlier channels kept in FP16; rest quantized to INT8. Nearest quantization level — expected smallest degradation. |
| **C — 4-bit NF4** | `nf4` | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)` | ~4.5 GB (single T4) | NormalFloat4 + double quantization. Largest precision reduction — expected largest degradation and strongest hypernym shift. |

**What is quantized in each arm:** only the LLM transformer blocks (`model.model.*`). The vision tower (`model.visual`), the visual-text merger, and `lm_head` remain FP16 in all three arms. This is a hard invariant (see §9, rule 1).

**Optional arm D** (Phase 5, heavyweight): rebuild the reference index using the quantized LLM, then compare Q-visual vs Q-index. This tests whether quantization causes a coherent space shift (preserved if both sides shift) vs. genuine information loss (degraded even when both sides are quantized). Marked optional; skippable without invalidating H1/H2.

---

## 2. Locked design decisions (non-arm)

| Decision | Value | Rationale |
|---|---|---|
| VLM | `Qwen/Qwen2-VL-7B-Instruct` | Paper-exact model (paper Fig. 5); pre-computed VG index exists |
| Reference index | `McGill-NLP/latentlens-qwen2vl-embeddings` | Paper-exact VG corpus, this model's backbone |
| Index stays FP **always** | yes | Fixed yardstick: isolates the effect of quantization on *visual tokens*, not on the reference space |
| Conditions | `fp16`, `int8`, `nf4` | bitsandbytes, the "easiest" path; agreed |
| Quantization scope | **LLM backbone only** | Hypothesis is *LLM* quantization. Vision tower + merger + `lm_head` stay FP16 |
| H1 labeling | Paper's VLM judge (GPT-5 / GPT-4o), API | Assume API available; same as paper |
| H2 scorer | WordNet `min_depth` via NLTK | Fully automated, no API |
| Test images | PixMo-Cap validation set | Paper parity. Fallback: COCO val2017 |
| Patches per image | 1 (default) | Matches paper's 100 patches / 100 images |
| Reference impl to adapt | repo `quickstart.py` | Built around exactly this model+index pairing |
---

## 3. The reference index format (verified from the model card)

Each layer lives in its own directory: `layer_{L}/embeddings_cache.pt`, loadable with
`torch.load(..., weights_only=False)`. Contents per layer:

- `embeddings`: `torch.float16` tensor, shape `[300836, 3584]` (≈2.1 GB). Hidden dim 3584 = Qwen2-7B.
- `token_to_indices`: `dict[str, list[int]]` mapping a token string to its row indices in `embeddings`.
- `metadata`: `list[dict]` aligned to rows, each with the token string, token ID, **source caption**,
  and **position** of the token within that caption.

Available layers: `[1, 2, 4, 8, 16, 24, 26, 27]`. Total ≈17 GB for all 8.

Implications:
- For a neighbor at row `i` in layer `L`: `token_str = metadata_L[i]["token"]` (key name may vary —
  inspect the dict), and the **phrase context** is `metadata_L[i]["caption"]` with the matched token at
  `metadata_L[i]["position"]`. Use this to reconstruct full words for H2 (merge adjacent subwords).
- Keep `embeddings` on **CPU RAM** (Kaggle ≈30 GB; all 8 layers fit). Move a layer to GPU only for the
  matmul if convenient, then free it.

---

## 4. Environment

- Kaggle, accelerator **GPU T4 ×2 (32 GB total)** for the FP16 condition (7B FP16 ≈16.6 GB → needs
  `device_map="auto"` sharding across both cards). int8 (≈8 GB) and NF4 (≈4.5 GB) fit a single T4.
- Python deps: `torch`, `transformers` (build from source if Qwen2-VL errors — see model card),
  `accelerate`, `bitsandbytes`, `huggingface_hub`, `latentlens` (pip), `nltk` (+ `wordnet`, `omw-1.4`),
  `spacy` (+ `en_core_web_sm`), `wordfreq`, `scipy`, `statsmodels`, `numpy`, `pandas`, `matplotlib`,
  `pyyaml`, `tqdm`, `Pillow`.
- Determinism: set a global `SEED` (default 0); seed `random`, `numpy`, `torch`; deterministic image
  and patch sampling.

---

## 5. Repository layout

```
qllens/
  config.yaml
  requirements.txt
  src/
    config.py        # load/validate config.yaml -> dataclass
    data.py          # fetch + sample test images (PixMo-Cap val; COCO fallback)
    models.py        # load Qwen2-VL in fp16 / int8 / nf4 (LLM-only quant)
    index.py         # download + load VG embeddings_cache.pt; build searchable banks
    extract.py       # run VLM, locate visual tokens, dump hidden states + patch->bbox
    nn_search.py     # cosine top-5 vs reference layers -> Neighbor records
    judge.py         # H1 GPT judge (pluggable OpenAI-compatible client)
    wordnet_h2.py    # H2 min_depth, hypernym test, frequency control
    stats.py         # McNemar (H1), Wilcoxon (H2), bootstrap CIs
    plots.py         # H1/H2 layer curves, delta heatmaps
    run.py           # CLI orchestrator over phases
  cache/             # downloaded index, extracted FP visual tokens (reused by quant runs)
  data/              # downloaded test images + chosen patch metadata
  results/
    neighbors/       # per-condition JSONL of top-5 neighbors per (image,patch,layer)
    h1/              # judge labels + per-layer interpretability
    h2/              # depth scores
    stats/           # test outputs (json/csv)
    figures/         # plots
```

---

## 6. Config (`config.yaml`)

```yaml
seed: 0
model: "Qwen/Qwen2-VL-7B-Instruct"
index_repo: "McGill-NLP/latentlens-qwen2vl-embeddings"

conditions: ["fp16", "int8", "nf4"]
quantize_only_llm: true        # MUST: skip vision tower, merger, lm_head

# Visual-token layers we extract & analyze (paper's Qwen2-VL set)
visual_layers: [0, 1, 2, 4, 8, 16, 24, 26, 27]
# Reference index layers loaded for NN search (subset of available [1,2,4,8,16,24,26,27])
reference_layers_smoke: [8, 27]
reference_layers_full:  [1, 2, 4, 8, 16, 24, 26, 27]

top_k: 5
patches_per_image: 1
test_images:
  source: "pixmo_cap_val"      # fallback: "coco_val2017"
  n_smoke: 5
  n_full: null                 # set after smoke test (25 / 50 / 100)

judge:
  enabled: true
  provider: "openai"           # pluggable
  model: "gpt-5"               # configurable; gpt-4o acceptable
  # prompt: paper Appendix C.1 (see judge.py)

paths:
  cache_dir: "cache"
  results_dir: "results"
```

---

## 7. Module contracts

### `models.py`
- `load_model(condition: str) -> (model, processor)`.
- `fp16`: `Qwen2VLForConditionalGeneration.from_pretrained(model, torch_dtype=fp16, device_map="auto")`.
- `int8` / `nf4`: `BitsAndBytesConfig`.
  - int8: `load_in_8bit=True`.
  - nf4: `load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16`.
  - **MUST** quantize only the language model. Inspect module names on the loaded model and pass
    the vision tower + merger + `lm_head` to `llm_int8_skip_modules` (the visual tower is typically
    `model.visual`; verify). Assert post-load that `model.visual` params are fp16 and LLM Linear
    layers are quantized; log the assertion.
- Always `model.eval()`, `torch.inference_mode()`.

### `index.py`
- `download_index(repo, layers) -> dict[int, Path]` via `hf_hub_download` of `layer_{L}/embeddings_cache.pt`.
- `load_banks(layers) -> dict[int, Bank]` where `Bank` holds the `[N,3584]` fp16 tensor (CPU),
  L2-normalized rows precomputed in fp32 for stable cosine, plus `metadata`.
- Banks are FP and **never** rebuilt for quantized conditions.

### `extract.py`
- `extract(model, processor, image, patch_indices) -> dict[layer -> Tensor[hidden_dim]]`.
- Format the input with `processor.apply_chat_template(...)` so image placeholder tokens are inserted
  (see `quickstart.py`).
- Run forward with `output_hidden_states=True`. `hidden_states[ℓ]` indexes layer ℓ (note: index 0 is
  the embedding output; map carefully so our `visual_layers` align with the index's layer numbering —
  document the mapping you choose).
- **Locate visual tokens**: find positions of the image pad token id between `vision_start`/`vision_end`.
  Qwen2-VL uses **spatial merge 2×2**, so one visual token ↔ a merged 2×2 patch block. Derive the patch
  grid from `image_grid_thw`.
- **Patch sampling**: deterministic given `seed`; default 1 patch/image. Save chosen patch indices.
- **Patch → pixel bbox** (needed by the judge): map the chosen visual-token index back to a pixel region
  on the processed image, accounting for the 2×2 merge and the grid from `image_grid_thw`. This is the
  fiddliest step — write a unit test that overlays the bbox on the image for visual sanity-checking.
- Save extracted hidden states to `cache/visual_tokens/{condition}/{image_id}.pt`. **FP tokens are
  extracted once and reused conceptually as the baseline**; quantized conditions re-extract.

### `nn_search.py`
- `search(query: Tensor[hidden_dim], banks, top_k) -> list[Neighbor]`.
- For each loaded reference layer L: `sims = normalize(query) @ banks[L].normed.T`; collect
  `(sim, layer=L, row)`. Take the **global top-k across all loaded layers**.
- `Neighbor = {token_str, similarity, contextual_layer, caption, position}` (caption/position from metadata).
- Output one JSONL row per `(condition, image_id, patch_idx, visual_layer)` with its top-5 Neighbors.

### `judge.py` (H1)
- Implements the paper's **Appendix C.1** judge (the user has the paper; use that exact prompt text).
  Pluggable client behind an interface `Judge.label(image, bbox, candidate_words) -> dict`.
- Judge inputs: the full image with the red bbox drawn, optionally the cropped region, and the
  **top-5 candidate words** for one visual token. Output JSON fields (per Appendix C.1):
  `reasoning` (str), `interpretable` (bool), `concrete_words` (list), `abstract_words` (list),
  `global_words` (list).
- A visual token is **interpretable** iff `interpretable == true` (≥1 of top-5 judged related).
- Only LatentLens descriptions are judged (we are not running EmbeddingLens/LogitLens).
- **Cost guard**: judging scales as `conditions × patches × visual_layers`. Log an estimated call
  count and approximate cost (~$1 / 100 calls order-of-magnitude) before running; support
  `--max-judge-calls` and resumable caching keyed by `(condition,image,patch,layer)`.

### `wordnet_h2.py` (H2)
- `word_min_depth(word) -> Optional[int]`: `min(s.min_depth() for s in wn.synsets(word, pos=wn.NOUN))` or `None`.
- For each neighbor: reconstruct the **full word** from `token_str` using `caption` + `position`
  (merge adjacent subwords, as in the paper); POS-tag with spaCy; keep nouns.
- Per-token H2 score = mean of available top-5 `min_depth` values (skip `None`); record `n_valid`.
- **Direct hypernym test** (stronger form of H2, no frequency confound): for the same (image,patch,layer),
  take FP top-1 noun and Q top-1 noun; report whether the Q word is a WordNet **ancestor (hypernym)**
  of the FP word, the reverse, or unrelated. Aggregate the asymmetry.
- **Frequency control** (optional): regress `min_depth` on `log10(wordfreq.word_frequency(word,'en'))`;
  report whether the FP→Q depth shift survives controlling for frequency.

### `stats.py`
- **H1**: per visual layer, `% interpretable` for each condition. Paired **McNemar** test on the same
  tokens (FP vs int8; FP vs nf4) per layer; report Δ% with bootstrap 95% CIs. Headline view: three
  curves over `visual_layers`, plus a Δ(condition−FP) curve to read the "later-layer widening".
- **H2**: per visual layer, paired **Wilcoxon signed-rank** on per-token mean `min_depth` (FP vs int8;
  FP vs nf4). Report median Δdepth + effect size (rank-biserial). Plus the hypernym-test asymmetry per layer.
- Save tidy CSVs and a `summary.json` stating, per hypothesis, whether the predicted direction holds
  and where (which layers).

### `plots.py`
- H1: % interpretable vs layer (3 curves) + Δ-vs-FP curve.
- H2: mean `min_depth` vs layer (3 curves) + Δ-vs-FP curve; hypernym-asymmetry bar per layer.
- Optional: heatmap of Δ across `condition × layer` for both metrics.

---

## 8. Execution phases (`run.py`)

- **Phase 0 — Smoke (single T4, no judge unless cheap).** `n_smoke=5`, `reference_layers_smoke`,
  conditions `["fp16"]` only. Goal: validate the full path end-to-end — load index, extract visual
  tokens, locate the patch, draw+verify the bbox, run NN search, run WordNet scoring. Confirm neighbor
  token strings look sane and `min_depth` values are populated. **Do not proceed until the bbox overlay
  looks correct.**
- **Phase 1 — FP baseline (Kaggle 2×T4).** `condition=fp16`, full `n_full` (set after smoke),
  `reference_layers_full`. Extract + NN search; cache FP neighbors and visual tokens.
- **Phase 2 — Quantized (single T4).** `int8` then `nf4`. Re-extract visual tokens, NN-search against
  the **same FP banks**. Cache neighbors.
- **Phase 3 — H1 judge (API).** Judge LatentLens top-5 for all conditions/layers. Respect cost guard +
  cache.
- **Phase 4 — Analysis.** Compute H1 & H2 stats, write CSVs + `summary.json` + figures.
- **Phase 5 — Optional robustness (Condition C′).** Re-build the index with the **quantized** model
  (`latentlens.build_index` over the same VG phrases, or the repo's extract pipeline) and re-run NN search
  Q-visual-vs-Q-index. If interpretability is preserved here but degraded vs the FP index, the effect is a
  coherent re-alignable shift rather than information loss. Mark clearly optional; it is the heaviest phase.

Each phase MUST be independently runnable and resumable (`run.py --phase N`), reading caches from prior phases.

---

## 9. Outputs

- `results/neighbors/{condition}.jsonl` — top-5 Neighbors per (image,patch,visual_layer).
- `results/h1/labels.jsonl` + `results/h1/interpretable_by_layer.csv`.
- `results/h2/depth_scores.csv` + `results/h2/hypernym_test.csv`.
- `results/stats/{h1_mcnemar,h2_wilcoxon}.csv`, `results/stats/summary.json`.
- `results/figures/*.png`.

---

## 10. Invariants & pitfalls (MUST respect)

1. **Quantize the LLM only.** Vision tower + merger + `lm_head` stay FP16. Assert this after load.
2. **The reference index is FP and identical across all conditions.** Never rebuild it for int8/nf4
   (except the explicitly optional Phase 5).
3. **Identical `reference_layers` across conditions** within a run, or H1/H2 comparisons are invalid.
4. **Hidden-state layer indexing**: be explicit about whether `hidden_states[i]` includes the embedding
   layer; align our `visual_layers` to the index's layer numbering and document it once.
5. **Patch→bbox** for Qwen2-VL dynamic resolution + 2×2 merge is error-prone — unit-test with an overlay.
6. **H2 word reconstruction & filtering**: VG vocabulary has many proper nouns / OOV tokens; expect a
   meaningful fraction of neighbors to drop out as non-nouns or `None`. Record `n_valid` per token and
   report coverage. Do not silently treat `None` as depth 0.
7. **Pairing for stats**: compare the *same* (image,patch,layer) across conditions; use paired tests.
8. **Do not compare absolute numbers to the paper** — only internal FP-vs-quantized deltas.
9. **Judge cost** is the only real money; cache aggressively and gate with `--max-judge-calls`.
10. **Determinism**: same patches and same image order across conditions (seeded).

---

## 11. Definition of done

- Phase 0 overlay confirms correct patch localization.
- For `n_full` images, all three conditions produce neighbor JSONLs, H1 labels, and H2 scores.
- `summary.json` reports, per hypothesis and per layer, the observed direction and significance:
  - H1: is `%interp(nf4) < %interp(int8) ≤ %interp(fp16)`, and does the FP gap widen in later layers (McNemar-significant where)?
  - H2: is median `min_depth(Q) < min_depth(FP)` (Wilcoxon-significant where), and does the direct hypernym test show Q-words are ancestors of FP-words more often than the reverse?
- Figures rendered for both hypotheses.