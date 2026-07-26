from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.io import read_json_object

CURRENT_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class VersionedConfig:
    kind: str
    schema_version: str
    values: dict[str, Any]


def load_versioned_config(path: Path, expected_kind: str) -> VersionedConfig:
    values = read_json_object(path)
    version = str(values.get("schema_version", CURRENT_SCHEMA_VERSION))
    kind = str(values.get("kind", expected_kind))
    if version != CURRENT_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Unsupported schema_version '{version}'; expected '{CURRENT_SCHEMA_VERSION}'"
        )
    if kind != expected_kind:
        raise ConfigurationError(f"Expected configuration kind '{expected_kind}', got '{kind}'")
    return VersionedConfig(kind=kind, schema_version=version, values=values)
