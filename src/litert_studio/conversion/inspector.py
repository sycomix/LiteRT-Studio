from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_studio.core.errors import InspectionError


@dataclass(frozen=True)
class ModelInspection:
    root: Path
    model_type: str
    architectures: tuple[str, ...]
    shard_files: tuple[str, ...]
    tensor_count: int | None
    parameter_count: int
    total_bytes: int
    fingerprint: str
    tokenizer_assets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "shard_files": list(self.shard_files),
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "weight_gib": round(self.total_bytes / (1024**3), 3),
            "estimated_training_memory_gib": {
                "lora_bf16": round(self.total_bytes * 2.5 / (1024**3), 2),
                "qlora_4bit": round(self.total_bytes * 0.8 / (1024**3), 2),
                "full_bf16_adamw": round(self.total_bytes * 8 / (1024**3), 2),
            },
            "total_bytes": self.total_bytes,
            "fingerprint": self.fingerprint,
            "tokenizer_assets": list(self.tokenizer_assets),
        }


def inspect_model_directory(root: Path) -> ModelInspection:
    root = root.resolve()
    if not root.is_dir():
        raise InspectionError(f"Model source is not a directory: {root}")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise InspectionError(f"Missing model configuration: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InspectionError(f"Invalid model config JSON: {exc}") from exc

    model_type = str(config.get("model_type", "unknown"))
    architectures = tuple(str(item) for item in config.get("architectures", []))
    shard_names, indexed_tensor_count = _resolve_shards(root)
    if not shard_names:
        raise InspectionError(f"No .safetensors files found in {root}")

    missing = [name for name in shard_names if not (root / name).is_file()]
    if missing:
        raise InspectionError(f"SafeTensors index references missing shards: {', '.join(missing)}")

    tokenizer_names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    tokenizer_assets = tuple(name for name in tokenizer_names if (root / name).is_file())
    total_bytes = sum((root / name).stat().st_size for name in shard_names)
    tensor_count, parameter_count = _safetensors_summary(root, shard_names)
    if indexed_tensor_count is not None and indexed_tensor_count != tensor_count:
        raise InspectionError("SafeTensors index tensor count does not match shard headers")
    fingerprint = _metadata_fingerprint(config_path, root, shard_names)
    return ModelInspection(
        root=root,
        model_type=model_type,
        architectures=architectures,
        shard_files=tuple(shard_names),
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        total_bytes=total_bytes,
        fingerprint=fingerprint,
        tokenizer_assets=tokenizer_assets,
    )


def _resolve_shards(root: Path) -> tuple[list[str], int | None]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise InspectionError(f"Invalid SafeTensors index: {index_path}") from exc
        if not isinstance(weight_map, dict):
            raise InspectionError("SafeTensors index 'weight_map' must be an object")
        return sorted({str(value) for value in weight_map.values()}), len(weight_map)
    return sorted(path.name for path in root.glob("*.safetensors")), None


def _metadata_fingerprint(config_path: Path, root: Path, shards: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    for name in shards:
        stat = (root / name).stat()
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return f"sha256-meta:{digest.hexdigest()}"


def _safetensors_summary(root: Path, shards: list[str]) -> tuple[int, int]:
    tensor_count = 0
    parameter_count = 0
    for name in shards:
        path = root / name
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
        for tensor_name, metadata in header.items():
            if tensor_name == "__metadata__":
                continue
            if not isinstance(metadata, dict) or not isinstance(metadata.get("shape"), list):
                raise InspectionError(f"Invalid tensor metadata '{tensor_name}' in {path}")
            parameters = 1
            for dimension in metadata["shape"]:
                if not isinstance(dimension, int) or dimension < 0:
                    raise InspectionError(f"Invalid tensor shape '{tensor_name}' in {path}")
                parameters *= dimension
            tensor_count += 1
            parameter_count += parameters
    return tensor_count, parameter_count
