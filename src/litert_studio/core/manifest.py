from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_type: str
    source_fingerprints: dict[str, str]
    files: tuple[ArtifactFile, ...]
    tools: dict[str, str]
    settings: dict[str, Any]
    validation: dict[str, Any]
    licenses: dict[str, str] = field(default_factory=dict)
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path, *, overwrite: bool = False) -> Path:
        if path.exists() and not overwrite:
            raise ConfigurationError(f"Manifest already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def describe_artifact(path: Path, *, relative_to: Path | None = None) -> ArtifactFile:
    path = path.resolve()
    if not path.is_file():
        raise ConfigurationError(f"Artifact file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    display_path = (
        str(path.relative_to(relative_to.resolve())) if relative_to is not None else path.name
    )
    return ArtifactFile(path=display_path, bytes=path.stat().st_size, sha256=digest.hexdigest())
