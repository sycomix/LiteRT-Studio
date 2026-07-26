from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_studio.core.config import load_versioned_config
from litert_studio.core.errors import ConfigurationError


def test_loads_matching_versioned_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"schema_version": "1", "kind": "conversion"}),
        encoding="utf-8",
    )
    assert load_versioned_config(path, "conversion").schema_version == "1"


def test_rejects_future_schema(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"schema_version": "99", "kind": "conversion"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Unsupported schema_version"):
        load_versioned_config(path, "conversion")
