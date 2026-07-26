from __future__ import annotations

from pathlib import Path
from typing import Any

from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.core.errors import ConfigurationError
from litert_studio.core.models import JobKind, JobPlan, Stage
from litert_studio.training.datasets import inspect_jsonl

METHODS = {"lora", "qlora", "full"}


def build_training_plan(config: dict[str, Any]) -> JobPlan:
    base_model = Path(_string(config, "base_model"))
    dataset = Path(_string(config, "dataset")).resolve()
    output = Path(_string(config, "output")).resolve()
    method = str(config.get("method", "lora"))
    if method not in METHODS:
        raise ConfigurationError(f"'method' must be one of: {', '.join(sorted(METHODS))}")
    if not dataset.is_file():
        raise ConfigurationError(f"Dataset does not exist: {dataset}")

    model = inspect_model_directory(base_model)
    dataset_inspection = inspect_jsonl(
        dataset,
        int(config.get("validation_sample_size", 100)),
    )
    epochs = _positive_number(config.get("epochs", 1), "epochs")
    batch_size = _positive_int(config.get("batch_size", 1), "batch_size")
    accumulation = _positive_int(
        config.get("gradient_accumulation_steps", 8),
        "gradient_accumulation_steps",
    )
    max_length = _positive_int(config.get("max_sequence_length", 2048), "max_sequence_length")

    warnings: list[str] = []
    if not config.get("eval_dataset") and not config.get("validation_split"):
        warnings.append("No evaluation dataset or validation split is configured.")
    if method == "full":
        warnings.append("Full fine-tuning may require substantially more accelerator memory.")

    recipe = {
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "max_sequence_length": max_length,
        "learning_rate": config.get("learning_rate", 0.0002),
        "seed": config.get("seed", 42),
        "precision": config.get("precision", "bf16"),
        "lora": config.get(
            "lora",
            {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": "auto"},
        ),
    }
    stages = (
        Stage("inspect", "Validate model, dataset, and licensing metadata", "studio-core"),
        Stage("prepare-data", "Format, tokenize, split, and fingerprint data", "transformers"),
        Stage("estimate", "Estimate memory and resolve an execution runner", "studio-estimator"),
        Stage("train", "Run the configured fine-tuning recipe", "transformers-peft", recipe),
        Stage("evaluate", "Evaluate checkpoints on the pinned suite", "studio-evaluator"),
        Stage(
            "save",
            "Save adapter or merged SafeTensors with provenance",
            "safetensors",
            {"merge_adapter": bool(config.get("merge_adapter", False))},
        ),
    )
    return JobPlan(
        kind=JobKind.TRAINING,
        name=str(config.get("name", f"train-{model.model_type}-{method}")),
        stages=stages,
        inputs={
            "base_model": model.to_dict(),
            "dataset": dataset_inspection.to_dict(),
        },
        outputs={"directory": str(output), "format": "safetensors"},
        warnings=tuple(warnings),
    )


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{key}' must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"'{name}' must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"'{name}' must be positive")
    return value
