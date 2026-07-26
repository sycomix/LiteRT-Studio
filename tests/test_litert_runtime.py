from __future__ import annotations

import json
from pathlib import Path

from litert_studio.conversion.litert_runtime import (
    LiteRTRuntimeReport,
    RuntimeCase,
    _record_runtime_result,
)
from litert_studio.conversion.validation import TokenParity


def test_runtime_report_requires_input_and_output_parity() -> None:
    case = RuntimeCase(
        prompt_sha256="a" * 64,
        input_tokens_match=True,
        token_parity=TokenParity(
            passed=True,
            reference_length=2,
            candidate_length=2,
            matching_prefix=2,
            exact_match=True,
        ),
        candidate_ids=(1, 2),
    )
    report = LiteRTRuntimeReport(
        passed=True,
        model_sha256="b" * 64,
        litert_lm_version="test",
        backend="cpu",
        logits_available=False,
        cases=(case,),
    )
    assert report.passed
    assert not report.logits_available


def test_runtime_result_updates_artifact_manifest(tmp_path: Path) -> None:
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"model")
    report_path = tmp_path / "runtime.json"
    report = LiteRTRuntimeReport(
        passed=False,
        model_sha256="b" * 64,
        litert_lm_version="test",
        backend="cpu",
        logits_available=False,
        cases=(),
    )
    report.write(report_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"validation": {"runtime": "pending"}}),
        encoding="utf-8",
    )

    _record_runtime_result(model, report_path, report)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["runtime"] == "failed"
    assert manifest["validation"]["runtime_report"]["backend"] == "cpu"
