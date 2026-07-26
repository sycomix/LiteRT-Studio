from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class DatasetInspection:
    path: Path
    fingerprint: str
    bytes: int
    records: int
    sampled_records: int
    formats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "bytes": self.bytes,
            "records": self.records,
            "sampled_records": self.sampled_records,
            "formats": list(self.formats),
        }


def inspect_jsonl(path: Path, sample_limit: int = 100) -> DatasetInspection:
    path = path.resolve()
    if sample_limit < 1:
        raise ConfigurationError("'validation_sample_size' must be positive")
    if not path.is_file():
        raise ConfigurationError(f"Dataset does not exist: {path}")

    digest = hashlib.sha256()
    records = 0
    sampled = 0
    formats: set[str] = set()
    with path.open("rb") as raw:
        for line_number, raw_line in enumerate(raw, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            records += 1
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ConfigurationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ConfigurationError(f"Record {line_number} must be a JSON object")
            record_format = (
                "text" if "text" in record else "messages" if "messages" in record else None
            )
            if record_format is None:
                raise ConfigurationError(
                    f"Record {line_number} must be an object with 'text' or 'messages'"
                )
            if sampled < sample_limit:
                formats.add(record_format)
                sampled += 1
    if records == 0:
        raise ConfigurationError(f"Dataset contains no records: {path}")
    return DatasetInspection(
        path=path,
        fingerprint=f"sha256:{digest.hexdigest()}",
        bytes=path.stat().st_size,
        records=records,
        sampled_records=sampled,
        formats=tuple(sorted(formats)),
    )
