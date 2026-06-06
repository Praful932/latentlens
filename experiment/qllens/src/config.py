from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    seed: int
    model: str
    index_dir: Path
    pixmo_cap_dir: Path
    conditions: list
    quantize_only_llm: bool
    visual_layers: list
    reference_layers_smoke: list
    reference_layers_full: list
    top_k: int
    patches_per_image: int
    # test_images flattened
    test_images_source: str
    test_images_n_smoke: int
    test_images_n_full: object  # int | None
    # judge flattened
    judge_enabled: bool
    judge_provider: str
    judge_model: str
    # resolved paths
    results_dir: Path
    cache_dir: Path


def load_config(path=None) -> Config:
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    path = Path(path).resolve()
    root = path.parent

    with open(path) as f:
        d = yaml.safe_load(f)

    def resolve(p):
        return (root / p).resolve()

    ti = d["test_images"]
    j = d["judge"]
    paths = d["paths"]

    raw_model = d["model"]
    # resolve local paths; leave HuggingFace hub IDs (no . or /) as-is
    model_val = str(resolve(raw_model)) if raw_model.startswith((".", "/")) else raw_model

    return Config(
        seed=d["seed"],
        model=model_val,
        index_dir=resolve(d["index_dir"]),
        pixmo_cap_dir=resolve(d["pixmo_cap_dir"]),
        conditions=d["conditions"],
        quantize_only_llm=d["quantize_only_llm"],
        visual_layers=d["visual_layers"],
        reference_layers_smoke=d["reference_layers_smoke"],
        reference_layers_full=d["reference_layers_full"],
        top_k=d["top_k"],
        patches_per_image=d["patches_per_image"],
        test_images_source=ti["source"],
        test_images_n_smoke=ti["n_smoke"],
        test_images_n_full=ti["n_full"],
        judge_enabled=j["enabled"],
        judge_provider=j["provider"],
        judge_model=j["model"],
        results_dir=resolve(paths["results_dir"]),
        cache_dir=resolve(paths["cache_dir"]),
    )
