from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.errors import InspectionError


def test_inspects_single_safetensors_model(model_dir: Path) -> None:
    result = inspect_model_directory(model_dir)
    assert result.model_type == "tiny"
    assert result.architectures == ("TinyForCausalLM",)
    assert result.shard_files == ("model.safetensors",)
    assert result.total_bytes > 4
    assert result.tensor_count == 1
    assert result.parameter_count == 1
    assert result.fingerprint.startswith("sha256-meta:")


def test_rejects_missing_index_shard(model_dir: Path) -> None:
    (model_dir / "model.safetensors").unlink()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": "missing-00001.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(InspectionError, match="missing shards"):
        inspect_model_directory(model_dir)
