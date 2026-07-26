from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import ArtifactManifest, describe_artifact
from litert_studio.core.packaging import create_bundle, verify_bundle


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()
    model = root / "model.litertlm"
    model.write_bytes(b"model payload")
    ArtifactManifest(
        artifact_type="litertlm",
        source_fingerprints={"model": "sha256:test"},
        files=(describe_artifact(model, relative_to=root),),
        tools={},
        settings={},
        validation={"runtime": "passed"},
    ).write(root / "manifest.json")
    return root


def test_bundle_is_reproducible_and_verifiable(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    report = tmp_path / "runtime-report.json"
    report.write_text('{"passed": true}\n', encoding="utf-8")

    first = create_bundle(root, tmp_path / "first.litertstudio", reports=(report,))
    second = create_bundle(root, tmp_path / "second.litertstudio", reports=(report,))

    assert first.sha256 == second.sha256
    assert verify_bundle(first.path).sha256 == first.sha256
    with zipfile.ZipFile(first.path) as archive:
        assert archive.namelist() == [
            "bundle.json",
            "artifacts/model.litertlm",
            "manifest.json",
            "reports/runtime-report.json",
        ]


def test_bundle_rejects_artifact_changed_after_manifest(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    (root / "model.litertlm").write_bytes(b"changed")

    with pytest.raises(ConfigurationError, match="mismatch"):
        create_bundle(root, tmp_path / "release.litertstudio")


def test_bundle_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../model.litertlm"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unsafe"):
        create_bundle(root, tmp_path / "release.litertstudio")
