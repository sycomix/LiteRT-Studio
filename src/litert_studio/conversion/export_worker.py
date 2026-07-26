from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from litert_studio.conversion.export_request import (
    LiteRTExportRequest,
    export_request_from_file,
)
from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.conversion.tensors import inspect_export_compatibility
from litert_studio.core.errors import ConfigurationError, StudioError
from litert_studio.core.manifest import ArtifactManifest, describe_artifact


@dataclass(frozen=True)
class ExportPreflight:
    ready: bool
    platform: str
    litert_torch_version: str | None
    mapping_compatible: bool
    issues: tuple[str, ...]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Isolated LiteRT Torch export worker")
    root.add_argument("--request", type=Path, required=True)
    root.add_argument("--execute", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        request = export_request_from_file(args.request)
        mapping = inspect_export_compatibility(Path(request.model))
        issues: list[str] = []
        system = platform.system().lower()
        if system != "linux":
            issues.append("LiteRT Torch generative export currently requires Linux")
        package_version = _version("litert-torch")
        if package_version is None:
            issues.append("litert-torch is not installed")
        if not mapping.compatible:
            issues.extend(mapping.issues or ("Model export compatibility audit did not pass",))
        report = ExportPreflight(
            ready=not issues,
            platform=system,
            litert_torch_version=package_version,
            mapping_compatible=mapping.compatible,
            issues=tuple(issues),
        )
        print(json.dumps(asdict(report), indent=2))
        if not args.execute:
            return 0 if report.ready else 2
        if not report.ready:
            return 2
        existing = existing_export(request)
        if existing is not None:
            print(
                json.dumps(
                    {
                        "status": "succeeded",
                        "request_id": request.request_id,
                        "artifact": str(existing),
                        "reused": True,
                    }
                )
            )
            return 0
        from litert_torch.generative.export_hf import (  # type: ignore[import-not-found]
            export as export_module,
        )

        export_module.export(
            model=request.model,
            output_dir=request.output_dir,
            task=request.task,
            trust_remote_code=False,
            prefill_lengths=list(request.prefill_lengths),
            cache_length=request.cache_length,
            # LiteRT Torch's public function drops None-valued arguments and then
            # applies its dynamic-int8 default. An explicit empty string selects
            # the unquantized baseline required for parity validation.
            quantization_recipe=request.quantization_recipe or "",
            externalize_embedder=request.externalize_embedder,
            use_jinja_template=request.use_jinja_template,
            bundle_litert_lm=request.bundle_litert_lm,
        )
        artifact, manifest = finalize_export(request)
        print(
            json.dumps(
                {
                    "status": "succeeded",
                    "request_id": request.request_id,
                    "artifact": str(artifact),
                    "manifest": str(manifest),
                    "reused": False,
                }
            )
        )
        return 0
    except (ConfigurationError, StudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def finalize_export(request: LiteRTExportRequest) -> tuple[Path, Path]:
    output = Path(request.output_dir)
    artifact = output / "model.litertlm"
    if not artifact.is_file():
        raise ConfigurationError(f"LiteRT-LM artifact was not produced: {artifact}")
    mapping = inspect_export_compatibility(Path(request.model))
    manifest = ArtifactManifest(
        artifact_type="litertlm",
        source_fingerprints={
            "export_request": request.request_id,
            "model": inspect_model_directory(Path(request.model)).fingerprint,
        },
        files=(describe_artifact(artifact, relative_to=output),),
        tools={
            name: found
            for name in (
                "litert-torch",
                "litert-lm-builder",
                "torch",
                "transformers",
            )
            if (found := _version(name)) is not None
        },
        settings=request.to_dict(),
        validation={
            "tensor_mapping_compatible": mapping.compatible,
            "tensor_count": mapping.tensor_count,
            "layer_count": mapping.layer_count,
            "validation_level": mapping.validation_level,
            "runtime": "pending",
        },
    )
    manifest_path = manifest.write(output / "manifest.json", overwrite=True)
    return artifact, manifest_path


def existing_export(request: LiteRTExportRequest) -> Path | None:
    output = Path(request.output_dir)
    artifact = output / "model.litertlm"
    manifest_path = output / "manifest.json"
    if not artifact.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    expected = {
        "export_request": request.request_id,
        "model": inspect_model_directory(Path(request.model)).fingerprint,
    }
    if manifest.get("source_fingerprints") != expected:
        return None
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return None
    recorded_hash = files[0].get("sha256")
    if recorded_hash != describe_artifact(artifact).sha256:
        return None
    return artifact


if __name__ == "__main__":
    raise SystemExit(main())
