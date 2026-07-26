import json
from pathlib import Path

from litert_studio.conversion.benchmark import _quantization_for_model


def test_quantization_is_read_from_export_manifest(tmp_path: Path) -> None:
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"model")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"settings": {"quantization_recipe": "dynamic_wi8_afp32"}}),
        encoding="utf-8",
    )

    assert _quantization_for_model(model) == "dynamic_int8"


def test_quantization_is_unknown_without_manifest(tmp_path: Path) -> None:
    assert _quantization_for_model(tmp_path / "model.litertlm") == "unknown"
