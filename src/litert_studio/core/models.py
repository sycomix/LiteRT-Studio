from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobKind(str, Enum):
    CONVERSION = "conversion"
    TRAINING = "training"


class JobState(str, Enum):
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    backend: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobPlan:
    kind: JobKind
    name: str
    stages: tuple[Stage, ...]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    warnings: tuple[str, ...] = ()
    schema_version: str = "1"

    @property
    def job_id(self) -> str:
        payload = json.dumps(self.to_dict(include_id=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        if include_id:
            value["job_id"] = self.job_id
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class Project:
    name: str
    root: Path
    schema_version: str = "1"

    def initialize(self) -> Path:
        metadata_dir = self.root / ".litert-studio"
        metadata_dir.mkdir(parents=True, exist_ok=False)
        for directory in ("plans", "runs", "artifacts"):
            (metadata_dir / directory).mkdir()
        project_file = metadata_dir / "project.json"
        project_file.write_text(
            json.dumps(
                {"name": self.name, "schema_version": self.schema_version},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return project_file
