from __future__ import annotations

from pathlib import Path

import pytest

from litert_studio.conversion import build_conversion_plan
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.models import JobKind


def test_conversion_plan_is_deterministic(model_dir: Path, tmp_path: Path) -> None:
    config = {
        "source": str(model_dir),
        "output": str(tmp_path / "out"),
        "quantization": "weight_only_int8",
    }
    first = build_conversion_plan(config)
    second = build_conversion_plan(config)
    assert first.job_id == second.job_id
    assert first.kind is JobKind.CONVERSION
    assert [stage.name for stage in first.stages] == [
        "inspect",
        "load",
        "transform",
        "export",
        "convert",
        "quantize",
        "verify",
        "package",
    ]


def test_static_int8_requires_separate_pipeline(model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="experimental calibration"):
        build_conversion_plan(
            {
                "source": str(model_dir),
                "output": str(tmp_path / "out"),
                "quantization": "static_int8",
            }
        )
