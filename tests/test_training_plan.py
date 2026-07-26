from __future__ import annotations

from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.training import build_training_plan


def test_training_plan_computes_effective_batch(
    model_dir: Path, dataset: Path, tmp_path: Path
) -> None:
    plan = build_training_plan(
        {
            "base_model": str(model_dir),
            "dataset": str(dataset),
            "output": str(tmp_path / "out"),
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "validation_split": 0.1,
        }
    )
    train = next(stage for stage in plan.stages if stage.name == "train")
    assert train.settings["effective_batch_size"] == 8
    assert plan.inputs["dataset"]["sampled_records"] == 2
    assert plan.inputs["dataset"]["fingerprint"].startswith("sha256:")


def test_training_rejects_invalid_record(model_dir: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text('{"prompt": "missing supported field"}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="'text' or 'messages'"):
        build_training_plan(
            {
                "base_model": str(model_dir),
                "dataset": str(dataset),
                "output": str(tmp_path / "out"),
            }
        )
