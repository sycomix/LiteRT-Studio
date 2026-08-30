from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litert_studio.core.errors import ConfigurationError

SUPPORTED_DATASET_EXTENSIONS = (".json", ".jsonl", ".ndjson", ".txt", ".text")


def dataset_record_format(record: dict[str, Any]) -> str | None:
    if "text" in record:
        return "text"
    if "messages" in record:
        return "messages"
    if "instruction" in record and "output" in record:
        return "instruction"
    if "prompt" in record and ("completion" in record or "response" in record):
        return "prompt_completion"
    return None


def value_at_field(record: dict[str, Any], field: str) -> Any:
    """Resolve a field name, including dotted paths such as ``payload.answer``."""
    value: Any = record
    for part in field.split("."):
        if not part or not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def normalize_dataset_fields(
    source: Path,
    destination: Path,
    mapping: dict[str, str],
) -> int:
    """Write a canonical JSONL copy using user-selected source fields."""
    allowed = {"text", "messages", "instruction", "input", "output", "prompt", "completion"}
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown dataset mapping targets: {', '.join(sorted(unknown))}")
    modes = sum(
        (
            bool(mapping.get("text")),
            bool(mapping.get("messages")),
            bool(mapping.get("instruction") or mapping.get("output")),
            bool(mapping.get("prompt") or mapping.get("completion")),
        )
    )
    if modes != 1:
        raise ConfigurationError("Map exactly one dataset shape")
    if bool(mapping.get("instruction")) != bool(mapping.get("output")):
        raise ConfigurationError("Instruction mappings require both instruction and output fields")
    if bool(mapping.get("prompt")) != bool(mapping.get("completion")):
        raise ConfigurationError("Prompt mappings require both prompt and completion fields")

    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record_number, record in iter_dataset_records(source):
                normalized = {
                    target: value_at_field(record, field)
                    for target, field in mapping.items()
                    if field
                }
                missing = [target for target, value in normalized.items() if value is None]
                if missing:
                    raise ConfigurationError(
                        f"Record {record_number} is missing mapped field(s): {', '.join(missing)}"
                    )
                handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
                count += 1
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if count == 0:
        destination.unlink(missing_ok=True)
        raise ConfigurationError(f"Dataset contains no records: {source}")
    return count


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


def iter_dataset_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        yield line_number, {"text": line.strip()}
        except UnicodeDecodeError as exc:
            raise ConfigurationError(f"Dataset is not valid UTF-8: {path}") from exc
        return
    if suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConfigurationError(f"Invalid JSON at {path}: {exc}") from exc
        if isinstance(value, dict):
            for key in ("records", "data", "train"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        records = value if isinstance(value, list) else [value]
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ConfigurationError(f"Record {index} must be a JSON object")
            yield index, record
        return
    if suffix not in {".jsonl", ".ndjson"}:
        supported = ", ".join(SUPPORTED_DATASET_EXTENSIONS)
        raise ConfigurationError(
            f"Unsupported dataset type '{suffix or '(none)'}'; use {supported}"
        )
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ConfigurationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ConfigurationError(f"Record {line_number} must be a JSON object")
            yield line_number, record


def inspect_dataset(path: Path, sample_limit: int = 100) -> DatasetInspection:
    path = path.resolve()
    if sample_limit < 1:
        raise ConfigurationError("'validation_sample_size' must be positive")
    if not path.is_file():
        raise ConfigurationError(f"Dataset does not exist: {path}")

    digest = hashlib.sha256(path.read_bytes())
    records = 0
    sampled = 0
    formats: set[str] = set()
    for line_number, record in iter_dataset_records(path):
        records += 1
        record_format = dataset_record_format(record)
        if record_format is None:
            raise ConfigurationError(
                f"Record {line_number} must contain 'text', 'messages', "
                "'instruction'/'output', or 'prompt'/'completion'"
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


def inspect_jsonl(path: Path, sample_limit: int = 100) -> DatasetInspection:
    """Compatibility alias for callers using the original JSONL-only API."""
    return inspect_dataset(path, sample_limit)
