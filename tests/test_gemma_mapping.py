from __future__ import annotations

import json
from pathlib import Path

from litert_studio.conversion.tensors import inspect_gemma_mapping, read_safetensors_metadata

SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * 4
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (8 - len(encoded) % 8) % 8
    encoded += b" " * padding
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * offset)


def _gemma_dir(tmp_path: Path) -> Path:
    model = tmp_path / "gemma"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma",
                "architectures": ["GemmaForCausalLM"],
                "num_hidden_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": ("F32", [16, 4]),
        "model.norm.weight": ("F32", [4]),
    }
    for suffix in SUFFIXES:
        rank = [4] if "layernorm" in suffix else [4, 4]
        tensors[f"model.layers.0.{suffix}"] = ("F32", rank)
    _write_safetensors(model / "model.safetensors", tensors)
    return model


def test_header_only_gemma_mapping_passes(tmp_path: Path) -> None:
    model = _gemma_dir(tmp_path)
    report = inspect_gemma_mapping(model)
    assert report.compatible
    assert report.layer_count == 1
    assert report.tensor_count == 11
    assert len(read_safetensors_metadata(model)) == 11


def test_mapping_reports_missing_tensor(tmp_path: Path) -> None:
    model = _gemma_dir(tmp_path)
    tensors = {
        tensor.name: (tensor.dtype, list(tensor.shape))
        for tensor in read_safetensors_metadata(model)
        if not tensor.name.endswith("q_proj.weight")
    }
    _write_safetensors(model / "model.safetensors", tensors)
    report = inspect_gemma_mapping(model)
    assert not report.compatible
    assert report.missing == ("model.layers.0.self_attn.q_proj.weight",)


def test_adapter_checkpoint_is_not_convertible_base(tmp_path: Path) -> None:
    model = _gemma_dir(tmp_path)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    report = inspect_gemma_mapping(model)
    assert report.adapter_checkpoint
    assert not report.compatible
