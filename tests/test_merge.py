from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import describe_artifact
from litert_studio.training.merge import _existing_merge, merge_adapter


def test_merge_refuses_to_overwrite_base(model_dir: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(ConfigurationError, match="must not overwrite"):
        merge_adapter(model_dir, adapter, model_dir)


def test_matching_completed_merge_is_reused(model_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "merged"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"merged")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    expected = {
        "base_model": inspect_model_directory(model_dir).fingerprint,
        "adapter": describe_artifact(adapter).sha256,
    }
    (output / "manifest.json").write_text(
        json.dumps({"source_fingerprints": expected}),
        encoding="utf-8",
    )
    result = _existing_merge(output, expected)
    assert result is not None
    assert result.reused
