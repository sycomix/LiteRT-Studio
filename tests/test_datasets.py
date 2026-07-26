from __future__ import annotations

from pathlib import Path

from litert_studio.training.datasets import inspect_jsonl


def test_dataset_fingerprint_changes_with_content(dataset: Path) -> None:
    first = inspect_jsonl(dataset)
    dataset.write_text('{"text": "different"}\n', encoding="utf-8")
    second = inspect_jsonl(dataset)
    assert first.fingerprint != second.fingerprint
    assert second.records == 1
    assert second.formats == ("text",)


def test_dataset_inspection_does_not_expose_records(dataset: Path) -> None:
    inspection = inspect_jsonl(dataset).to_dict()
    assert "hello" not in str(inspection)
    assert "messages" in inspection["formats"]
