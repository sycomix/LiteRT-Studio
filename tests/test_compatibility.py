from pathlib import Path

from litert_studio.core.compatibility import CompatibilityRegistry


def test_registry_records_and_lists_results(tmp_path: Path) -> None:
    registry = CompatibilityRegistry(tmp_path / "compatibility.sqlite3")
    created = registry.record(
        model_sha256="abc",
        quantization="dynamic_int8",
        runtime="litert-lm 0.14.0",
        device="Linux x86_64",
        result_type="benchmark",
        passed=True,
        report_path=tmp_path / "report.json",
        summary={"output_tokens_per_second_median": 12.5},
    )

    listed = registry.list_recent()
    assert listed == (created,)
    assert listed[0].summary["output_tokens_per_second_median"] == 12.5
