from __future__ import annotations

import json
from pathlib import Path

from litert_studio.conversion.adapters import AdapterRegistry
from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.conversion.tensors import inspect_export_compatibility


def test_qwen3_uses_upstream_extension_compatibility(tmp_path: Path) -> None:
    model = tmp_path / "qwen3"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "num_hidden_layers": 2,
            }
        ),
        encoding="utf-8",
    )
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "F32", "shape": [4, 4]}}).encode()
    (model / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + bytes(64)
    )

    inspection = inspect_model_directory(model)
    report = inspect_export_compatibility(model)

    assert AdapterRegistry().resolve(inspection) is not None
    assert report.compatible
    assert report.validation_level == "upstream_extension"
