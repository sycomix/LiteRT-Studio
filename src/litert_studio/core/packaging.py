from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import describe_artifact

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BundleResult:
    path: Path
    sha256: str
    files: int


def create_bundle(
    artifact_directory: Path,
    output: Path,
    *,
    reports: tuple[Path, ...] = (),
) -> BundleResult:
    root = artifact_directory.resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    entries: dict[str, Path] = {"manifest.json": manifest_path}
    for item in _manifest_files(manifest):
        relative = _safe_relative_path(item["path"])
        source = (root / Path(*relative.parts)).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ConfigurationError(f"Manifest artifact is unavailable: {relative}")
        described = describe_artifact(source)
        if described.bytes != item["bytes"] or described.sha256 != item["sha256"]:
            raise ConfigurationError(f"Manifest hash or size mismatch: {relative}")
        entries[f"artifacts/{relative.as_posix()}"] = source
    for report in reports:
        source = report.resolve()
        if not source.is_file():
            raise ConfigurationError(f"Validation report does not exist: {source}")
        archive_name = f"reports/{source.name}"
        if archive_name in entries:
            raise ConfigurationError(f"Duplicate bundle entry: {archive_name}")
        entries[archive_name] = source

    bundle_files = [
        {
            "path": name,
            "bytes": source.stat().st_size,
            "sha256": describe_artifact(source).sha256,
        }
        for name, source in sorted(entries.items())
    ]
    bundle_manifest = {
        "schema_version": "1",
        "format": "litert-studio-bundle",
        "source_manifest_sha256": describe_artifact(manifest_path).sha256,
        "files": bundle_files,
    }
    bundle_bytes = (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode()
    if output.exists():
        raise ConfigurationError(f"Bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        _write_bytes(archive, "bundle.json", bundle_bytes)
        for name, source in sorted(entries.items()):
            _write_bytes(archive, name, source.read_bytes())
    return BundleResult(
        path=output.resolve(),
        sha256=describe_artifact(output).sha256,
        files=len(entries) + 1,
    )


def verify_bundle(path: Path) -> BundleResult:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "bundle.json" not in names:
                raise ConfigurationError("Bundle has duplicate entries or no bundle.json")
            manifest = json.loads(archive.read("bundle.json"))
            if (
                manifest.get("schema_version") != "1"
                or manifest.get("format") != "litert-studio-bundle"
            ):
                raise ConfigurationError("Unsupported bundle format")
            expected_names = {"bundle.json"}
            for item in manifest["files"]:
                name = _safe_relative_path(item["path"]).as_posix()
                payload = archive.read(name)
                if len(payload) != item["bytes"] or _sha256(payload) != item["sha256"]:
                    raise ConfigurationError(f"Bundle hash or size mismatch: {name}")
                expected_names.add(name)
            if set(names) != expected_names:
                raise ConfigurationError("Bundle contains unmanifested files")
            if _sha256(archive.read("manifest.json")) != manifest.get("source_manifest_sha256"):
                raise ConfigurationError("Source manifest hash mismatch")
    except (
        FileNotFoundError,
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ConfigurationError(f"Invalid LiteRT Studio bundle: {path}") from exc
    return BundleResult(
        path=path.resolve(),
        sha256=describe_artifact(path).sha256,
        files=len(expected_names),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Invalid artifact manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Invalid artifact manifest: {path}")
    return value


def _manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ConfigurationError("Artifact manifest has no files")
    if not all(
        isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("bytes"), int)
        and isinstance(item.get("sha256"), str)
        for item in files
    ):
        raise ConfigurationError("Artifact manifest has invalid file entries")
    return files


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigurationError(f"Unsafe bundle path: {value}")
    return path


def _write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, _ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
