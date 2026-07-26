from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import ArtifactManifest, describe_artifact


def test_manifest_hashes_artifact_and_refuses_overwrite(tmp_path: Path) -> None:
    artifact = tmp_path / "model.litertlm"
    artifact.write_bytes(b"model")
    described = describe_artifact(artifact, relative_to=tmp_path)
    manifest = ArtifactManifest(
        artifact_type="litertlm",
        source_fingerprints={"model": "sha256:test"},
        files=(described,),
        tools={"litert-torch": "test"},
        settings={},
        validation={"passed": True},
    )
    output = manifest.write(tmp_path / "manifest.json")
    content = json.loads(output.read_text(encoding="utf-8"))
    assert content["files"][0]["path"] == "model.litertlm"
    assert len(content["files"][0]["sha256"]) == 64
    with pytest.raises(ConfigurationError, match="already exists"):
        manifest.write(output)
