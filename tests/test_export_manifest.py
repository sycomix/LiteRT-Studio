from __future__ import annotations

import json
from pathlib import Path

from test_gemma_mapping import _gemma_dir

from litert_studio.conversion.export_request import LiteRTExportRequest
from litert_studio.conversion.export_worker import existing_export, finalize_export
from litert_studio.core.manifest import describe_artifact


def test_finalize_and_reuse_export(tmp_path: Path, monkeypatch) -> None:
    model = _gemma_dir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "model.litertlm"
    artifact.write_bytes(b"litertlm")
    request = LiteRTExportRequest(model=str(model), output_dir=str(output))
    versions = {
        "litert-torch": "test",
        "litert-lm-builder": "test",
        "torch": "test",
        "transformers": "test",
    }
    monkeypatch.setattr(
        "litert_studio.conversion.export_worker._version",
        lambda name: versions.get(name),
    )
    produced, manifest_path = finalize_export(request)
    assert produced == artifact
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_fingerprints"]["export_request"] == request.request_id
    assert manifest["files"][0]["sha256"] == describe_artifact(artifact).sha256
    assert existing_export(request) == artifact


def test_modified_artifact_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    model = _gemma_dir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "model.litertlm"
    artifact.write_bytes(b"first")
    request = LiteRTExportRequest(model=str(model), output_dir=str(output))
    monkeypatch.setattr(
        "litert_studio.conversion.export_worker._version",
        lambda name: "test",
    )
    finalize_export(request)
    artifact.write_bytes(b"changed")
    assert existing_export(request) is None
