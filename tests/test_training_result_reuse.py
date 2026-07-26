from __future__ import annotations

import json
from pathlib import Path

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.training.datasets import inspect_jsonl
from litert_studio.training.recipe import recipe_from_config
from litert_studio.training.transformers_backend import _existing_result


def test_completed_matching_result_is_reused(
    model_dir: Path, dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    recipe = recipe_from_config(
        {
            "base_model": str(model_dir),
            "dataset": str(dataset),
            "output": str(output),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_fingerprints": {
                    "training_request": recipe.request_id,
                    "base_model": inspect_model_directory(model_dir).fingerprint,
                    "dataset": inspect_jsonl(dataset).fingerprint,
                },
                "settings": {"request_id": recipe.request_id},
                "validation": {"training_metrics": {"train_loss": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    result = _existing_result(recipe)
    assert result is not None
    assert result.reused
    assert result.metrics["train_loss"] == 1.0


def test_changed_dataset_invalidates_completed_result(
    model_dir: Path, dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    recipe = recipe_from_config(
        {
            "base_model": str(model_dir),
            "dataset": str(dataset),
            "output": str(output),
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_fingerprints": {
                    "training_request": recipe.request_id,
                    "base_model": inspect_model_directory(model_dir).fingerprint,
                    "dataset": "sha256:stale",
                },
                "settings": {"request_id": recipe.request_id},
                "validation": {"training_metrics": {}},
            }
        ),
        encoding="utf-8",
    )
    assert _existing_result(recipe) is None
