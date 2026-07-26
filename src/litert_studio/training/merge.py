from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.manifest import ArtifactManifest, describe_artifact


@dataclass(frozen=True)
class MergeResult:
    model_directory: Path
    manifest: Path
    reused: bool = False


def merge_adapter(base_model: Path, adapter: Path, output: Path) -> MergeResult:
    base_model = base_model.resolve()
    adapter = adapter.resolve()
    output = output.resolve()
    if output in {base_model, adapter}:
        raise ConfigurationError("Merged output must not overwrite the base model or adapter")
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise ConfigurationError(f"Adapter SafeTensors not found: {adapter_weights}")
    expected = {
        "base_model": inspect_model_directory(base_model).fingerprint,
        "adapter": describe_artifact(adapter_weights).sha256,
    }
    existing = _existing_merge(output, expected)
    if existing is not None:
        return existing

    os.environ.setdefault("USE_TF", "0")
    import torch
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    peft_model = PeftModel.from_pretrained(model, adapter)
    merged = peft_model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=False).save_pretrained(output)

    manifest = ArtifactManifest(
        artifact_type="merged-safetensors-model",
        source_fingerprints=expected,
        files=tuple(
            describe_artifact(path, relative_to=output)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ),
        tools={
            "torch": version("torch"),
            "transformers": version("transformers"),
            "peft": version("peft"),
            "safetensors": version("safetensors"),
        },
        settings={"safe_merge": True, "dtype": "float32"},
        validation={"tensor_layout": "pending-audit"},
    )
    manifest_path = manifest.write(output / "manifest.json", overwrite=True)
    return MergeResult(model_directory=output, manifest=manifest_path)


def _existing_merge(
    output: Path,
    expected_fingerprints: dict[str, str],
) -> MergeResult | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file() or not (output / "model.safetensors").is_file():
        return None
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("source_fingerprints") != expected_fingerprints:
        return None
    return MergeResult(model_directory=output, manifest=manifest_path, reused=True)
