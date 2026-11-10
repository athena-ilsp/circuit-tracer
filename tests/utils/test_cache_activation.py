"""A transcoder set loaded from the local cache keeps the activation its config.yaml names."""

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

from circuit_tracer.transcoder.activation_functions import TopK
from circuit_tracer.utils.caching import load_transcoders_from_cache

D_MODEL, D_SAE, N_LAYERS = 4, 8, 2


def _write_cache(cache_dir: Path, ref: str, extra: dict) -> None:
    path = cache_dir / ref
    path.mkdir(parents=True)
    config = {
        "model_kind": "transcoder_set",
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "mlp.hook_out",
        **extra,
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config))
    for layer in range(N_LAYERS):
        save_file(
            {
                "W_enc": torch.randn(D_SAE, D_MODEL),
                "W_dec": torch.randn(D_SAE, D_MODEL),
                "b_enc": torch.zeros(D_SAE),
                "b_dec": torch.zeros(D_MODEL),
            },
            str(path / f"layer_{layer}.safetensors"),
        )


def _activations(cache_dir: Path, ref: str) -> list:
    transcoders, _ = load_transcoders_from_cache(
        ref, cache_dir=cache_dir, device=torch.device("cpu")
    )
    return [transcoders[layer].activation_function for layer in range(N_LAYERS)]


def test_a_topk_relu_set_is_loaded_as_topk_relu(tmp_path: Path):
    _write_cache(tmp_path, "local/topk-relu", {"activation": "topk_relu", "k": 3})
    for fn in _activations(tmp_path, "local/topk-relu"):
        assert isinstance(fn, TopK) and fn.k == 3 and fn.relu


def test_a_topk_set_is_loaded_as_topk(tmp_path: Path):
    _write_cache(tmp_path, "local/topk", {"activation": "topk", "k": 2})
    for fn in _activations(tmp_path, "local/topk"):
        assert isinstance(fn, TopK) and fn.k == 2 and not fn.relu


def test_a_set_naming_no_activation_still_defaults_to_relu(tmp_path: Path):
    _write_cache(tmp_path, "local/plain", {})
    assert all(fn is F.relu for fn in _activations(tmp_path, "local/plain"))
