import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

FIXED_RESOLUTION = 448  # → 16×16 merged token grid (256 visual tokens)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_processor(cfg) -> AutoProcessor:
    """Load processor only — no model weights. Usable in phase 0a."""
    processor = AutoProcessor.from_pretrained(cfg.model)
    px = FIXED_RESOLUTION * FIXED_RESOLUTION
    processor.image_processor.min_pixels = px
    processor.image_processor.max_pixels = px
    # Prevent smart_resize from picking a different grid size
    processor.image_processor.do_resize = False
    return processor


def load_model(condition: str, cfg):
    """
    Load Qwen2-VL-7B-Instruct for the given condition.
    Phase 0 only uses 'fp16'. Returns (model, device).
    """
    device = get_device()

    if condition != "fp16":
        raise NotImplementedError(f"condition={condition!r} not implemented for Phase 0")

    if device.type == "cuda":
        # device_map="auto" shards across available GPUs (needed for 2×T4 in Phase 1)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        # device_map="auto" does not target MPS; fp16 on MPS can produce NaNs
        dtype = torch.float32 if device.type == "mps" else torch.float16
        if device.type == "mps":
            print("  Warning: using float32 on MPS (fp16 can NaN). Model will be slow.")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model, torch_dtype=dtype
        ).to(device)

    model.eval()
    return model, device
