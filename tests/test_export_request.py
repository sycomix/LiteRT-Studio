from __future__ import annotations

from pathlib import Path

from litert_studio.conversion.export_request import (
    export_request_from_config,
    export_request_from_file,
)


def test_export_request_round_trip(tmp_path: Path) -> None:
    request = export_request_from_config(
        {
            "source": str(tmp_path / "model"),
            "output": str(tmp_path / "output"),
            "prefill_lengths": [64, 128],
            "cache_length": 512,
            "quantization_recipe": "dynamic_wi8_afp32",
            "bundle_litert_lm": True,
        }
    )
    path = request.write(tmp_path / "request.json")
    loaded = export_request_from_file(path)
    assert loaded.request_id == request.request_id
    assert loaded.prefill_lengths == (64, 128)
    assert loaded.quantization_recipe == "dynamic_wi8_afp32"
