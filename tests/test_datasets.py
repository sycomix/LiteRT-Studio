from __future__ import annotations

from pathlib import Path

from litert_studio.training.datasets import inspect_jsonl, normalize_dataset_fields
from litert_studio.training.formatting import load_formatted_records


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


def test_json_array_and_plain_text_datasets_are_supported(tmp_path: Path) -> None:
    json_path = tmp_path / "custom.json"
    json_path.write_text('[{"text":"one"},{"text":"two"}]', encoding="utf-8")
    text_path = tmp_path / "custom.txt"
    text_path.write_text("first\n\nsecond\n", encoding="utf-8")

    assert inspect_jsonl(json_path).records == 2
    assert inspect_jsonl(text_path).records == 2
    assert load_formatted_records(json_path, object()) == ["one", "two"]
    assert load_formatted_records(text_path, object()) == ["first", "second"]


def test_alpaca_instruction_dataset_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "alpaca.json"
    path.write_text(
        '[{"instruction":"Optimize this","input":"constraints","output":"result"}]',
        encoding="utf-8",
    )

    inspection = inspect_jsonl(path)
    records = load_formatted_records(path, object())

    assert inspection.formats == ("instruction",)
    assert records == [
        "### Instruction:\nOptimize this\n\n### Input:\nconstraints\n\n### Response:\nresult"
    ]


def test_custom_nested_fields_can_be_normalized(tmp_path: Path) -> None:
    source = tmp_path / "custom.json"
    source.write_text(
        '[{"payload":{"question":"Why?","answer":"Because."},"context":"Reasoning"}]',
        encoding="utf-8",
    )
    normalized = tmp_path / "mapped.jsonl"

    count = normalize_dataset_fields(
        source,
        normalized,
        {"instruction": "payload.question", "input": "context", "output": "payload.answer"},
    )

    assert count == 1
    assert inspect_jsonl(normalized).formats == ("instruction",)
    assert "Because." in load_formatted_records(normalized, object())[0]
