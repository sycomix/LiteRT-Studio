from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from litert_studio.conversion.inspector import ModelInspection, inspect_model_directory
from litert_studio.core.errors import ConfigurationError

_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ImportedModel:
    repository: str
    requested_revision: str
    resolved_revision: str
    path: Path
    inspection: ModelInspection


def import_huggingface_model(
    repository: str,
    revision: str,
    destination_root: Path,
) -> ImportedModel:
    if not _REPOSITORY_ID.fullmatch(repository):
        raise ConfigurationError("Model repository must use the 'owner/name' form")
    if not revision.strip():
        raise ConfigurationError("A model revision or commit is required")
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ConfigurationError(
            "Install the training or conversion extra to import registry models"
        ) from exc
    try:
        info = HfApi().model_info(repository, revision=revision, files_metadata=False)
        resolved_revision = str(info.sha)
        destination = destination_root.resolve() / repository.replace("/", "--") / resolved_revision
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repository,
            revision=resolved_revision,
            local_dir=destination,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.safetensors",
                "*.safetensors.index.json",
                "LICENSE*",
                "README*",
            ],
        )
    except Exception as exc:
        raise ConfigurationError(
            f"Could not import '{repository}' at revision '{revision}': {exc}"
        ) from exc
    inspection = inspect_model_directory(destination)
    return ImportedModel(
        repository=repository,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        path=destination,
        inspection=inspection,
    )
