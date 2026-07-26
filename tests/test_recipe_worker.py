from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_studio.cli import main
from litert_studio.core.errors import ConfigurationError
from litert_studio.training.recipe import recipe_from_config, recipe_from_file
from litert_studio.training.worker import preflight


def test_recipe_round_trip_is_deterministic(model_dir: Path, dataset: Path, tmp_path: Path) -> None:
    config = {
        "base_model": str(model_dir),
        "dataset": str(dataset),
        "output": str(tmp_path / "output"),
        "method": "lora",
        "lora": {"target_modules": ["q_proj", "v_proj"]},
    }
    recipe = recipe_from_config(config)
    path = recipe.write(tmp_path / "request.json")
    loaded = recipe_from_file(path)
    assert loaded.request_id == recipe.request_id
    assert loaded.lora.target_modules == ("q_proj", "v_proj")


def test_preflight_reports_fingerprint_without_records(
    model_dir: Path, dataset: Path, tmp_path: Path
) -> None:
    recipe = recipe_from_config(
        {
            "base_model": str(model_dir),
            "dataset": str(dataset),
            "output": str(tmp_path / "output"),
        }
    )
    report = preflight(recipe)
    assert report.dataset_fingerprint.startswith("sha256:")
    assert "hello" not in json.dumps(report.to_dict())


def test_cli_materializes_training_request(
    model_dir: Path, dataset: Path, tmp_path: Path, capsys
) -> None:
    config = tmp_path / "training.json"
    request = tmp_path / "request.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "kind": "training",
                "base_model": str(model_dir),
                "dataset": str(dataset),
                "output": str(tmp_path / "output"),
            }
        ),
        encoding="utf-8",
    )
    assert main(["prepare-train", str(config), "--output", str(request)]) == 0
    assert request.is_file()
    assert "Wrote training request" in capsys.readouterr().out


def test_recipe_rejects_two_evaluation_strategies(
    model_dir: Path, dataset: Path, tmp_path: Path
) -> None:
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        recipe_from_config(
            {
                "base_model": str(model_dir),
                "dataset": str(dataset),
                "eval_dataset": str(dataset),
                "validation_split": 0.2,
                "output": str(tmp_path / "output"),
            }
        )
