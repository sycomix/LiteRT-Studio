from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.errors import InspectionError

_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
_LAYER_SUFFIXES = (
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


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str


@dataclass(frozen=True)
class GemmaMappingReport:
    compatible: bool
    model_type: str
    tensor_count: int
    layer_count: int
    expected_layer_count: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    invalid_ranks: tuple[str, ...]
    adapter_checkpoint: bool
    tensors: tuple[TensorMetadata, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExportCompatibilityReport:
    compatible: bool
    model_type: str
    tensor_count: int
    layer_count: int
    validation_level: str
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_UPSTREAM_MODEL_TYPES = {
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma4",
    "lfm2",
    "qwen3",
}


def inspect_export_compatibility(model_dir: Path) -> ExportCompatibilityReport:
    inspection = inspect_model_directory(model_dir)
    if inspection.model_type in {"gemma", "gemma2"}:
        mapping = inspect_gemma_mapping(model_dir)
        issues = (
            *mapping.missing,
            *mapping.unexpected,
            *mapping.invalid_ranks,
        )
        if mapping.adapter_checkpoint:
            issues = (*issues, "Checkpoint contains adapter tensors; merge it before export")
        return ExportCompatibilityReport(
            compatible=mapping.compatible,
            model_type=mapping.model_type,
            tensor_count=mapping.tensor_count,
            layer_count=mapping.layer_count,
            validation_level="reviewed_tensor_schema",
            issues=tuple(issues),
        )
    config = json.loads((inspection.root / "config.json").read_text(encoding="utf-8"))
    layer_count = int(config.get("num_hidden_layers", 0))
    supported = inspection.model_type in _UPSTREAM_MODEL_TYPES
    if not supported:
        return ExportCompatibilityReport(
            compatible=False,
            model_type=inspection.model_type,
            tensor_count=inspection.tensor_count or 0,
            layer_count=layer_count,
            validation_level="unsupported",
            issues=(
                f"LiteRT Torch 0.9 model extension is not registered for '{inspection.model_type}'",
            ),
        )
    tensors = read_safetensors_metadata(model_dir)
    return ExportCompatibilityReport(
        compatible=bool(tensors) and layer_count > 0,
        model_type=inspection.model_type,
        tensor_count=len(tensors),
        layer_count=layer_count,
        validation_level="upstream_extension",
        issues=(),
    )


def read_safetensors_metadata(model_dir: Path) -> tuple[TensorMetadata, ...]:
    inspection = inspect_model_directory(model_dir)
    tensors: list[TensorMetadata] = []
    for shard in inspection.shard_files:
        path = inspection.root / shard
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise InspectionError(f"Invalid SafeTensors header in {path}")
            header_length = int.from_bytes(length_bytes, "little")
            if header_length <= 0 or header_length > path.stat().st_size - 8:
                raise InspectionError(f"Invalid SafeTensors header length in {path}")
            try:
                header = json.loads(handle.read(header_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InspectionError(f"Invalid SafeTensors metadata in {path}") from exc
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise InspectionError(f"Invalid tensor entry '{name}' in {path}")
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                raise InspectionError(f"Invalid tensor metadata '{name}' in {path}")
            tensors.append(
                TensorMetadata(
                    name=name,
                    dtype=dtype,
                    shape=tuple(int(value) for value in shape),
                    shard=shard,
                )
            )
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise InspectionError("Duplicate tensor names found across SafeTensors shards")
    return tuple(sorted(tensors, key=lambda tensor: tensor.name))


def inspect_gemma_mapping(model_dir: Path) -> GemmaMappingReport:
    inspection = inspect_model_directory(model_dir)
    config_path = inspection.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_layers = int(config.get("num_hidden_layers", 0))
    tensors = read_safetensors_metadata(model_dir)
    names = {tensor.name for tensor in tensors}
    adapter_checkpoint = (inspection.root / "adapter_config.json").is_file() or any(
        "lora_" in name for name in names
    )

    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    for index in range(expected_layers):
        expected.update(f"model.layers.{index}.{suffix}" for suffix in _LAYER_SUFFIXES)
    missing = tuple(sorted(expected - names))

    recognized = set(expected)
    recognized.add("lm_head.weight")
    unexpected = tuple(sorted(names - recognized))
    invalid_ranks: list[str] = []
    for tensor in tensors:
        if tensor.name.endswith(("_proj.weight", "embed_tokens.weight", "lm_head.weight")):
            if len(tensor.shape) != 2:
                invalid_ranks.append(tensor.name)
        elif tensor.name.endswith("layernorm.weight") or tensor.name == "model.norm.weight":
            if len(tensor.shape) != 1:
                invalid_ranks.append(tensor.name)

    layers = {
        int(match.group(1)) for name in names if (match := _LAYER_PATTERN.match(name)) is not None
    }
    compatible_model_type = inspection.model_type in {"gemma", "gemma2"}
    compatible = (
        compatible_model_type
        and expected_layers > 0
        and layers == set(range(expected_layers))
        and not missing
        and not unexpected
        and not invalid_ranks
        and not adapter_checkpoint
    )
    return GemmaMappingReport(
        compatible=compatible,
        model_type=inspection.model_type,
        tensor_count=len(tensors),
        layer_count=len(layers),
        expected_layer_count=expected_layers,
        missing=missing,
        unexpected=unexpected,
        invalid_ranks=tuple(sorted(invalid_ranks)),
        adapter_checkpoint=adapter_checkpoint,
        tensors=tensors,
    )
