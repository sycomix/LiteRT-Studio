from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from litert_studio.core.errors import ConfigurationError


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | tuple[str, ...] = "auto"


@dataclass(frozen=True)
class TrainingRecipe:
    base_model: str
    dataset: str
    eval_dataset: str | None
    validation_split: float | None
    output: str
    method: str
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int
    learning_rate: float
    precision: str
    seed: int
    lora: LoraSettings
    max_steps: int = -1
    logging_steps: int = 1
    save_steps: int = 50
    resume_from_checkpoint: bool = True
    schema_version: str = "1"

    @property
    def request_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["request_id"] = self.request_id
        if isinstance(self.lora.target_modules, tuple):
            value["lora"]["target_modules"] = list(self.lora.target_modules)
        return value

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def recipe_from_config(config: dict[str, Any]) -> TrainingRecipe:
    lora_raw = config.get("lora", {})
    if not isinstance(lora_raw, dict):
        raise ConfigurationError("'lora' must be an object")
    targets = lora_raw.get("target_modules", "auto")
    if isinstance(targets, list):
        if not targets or not all(isinstance(item, str) and item for item in targets):
            raise ConfigurationError("'lora.target_modules' must contain non-empty strings")
        normalized_targets: str | tuple[str, ...] = tuple(targets)
    elif targets == "auto":
        normalized_targets = "auto"
    else:
        raise ConfigurationError("'lora.target_modules' must be 'auto' or a list of strings")

    eval_dataset = config.get("eval_dataset")
    validation_split = config.get("validation_split")
    if eval_dataset is not None and validation_split is not None:
        raise ConfigurationError("'eval_dataset' and 'validation_split' are mutually exclusive")
    if eval_dataset is not None and (not isinstance(eval_dataset, str) or not eval_dataset.strip()):
        raise ConfigurationError("'eval_dataset' must be a non-empty string")
    if validation_split is not None and (
        isinstance(validation_split, bool)
        or not isinstance(validation_split, (int, float))
        or not 0 < float(validation_split) < 1
    ):
        raise ConfigurationError("'validation_split' must be between 0 and 1")

    return TrainingRecipe(
        base_model=str(Path(_string(config, "base_model")).resolve()),
        dataset=str(Path(_string(config, "dataset")).resolve()),
        eval_dataset=(str(Path(eval_dataset).resolve()) if isinstance(eval_dataset, str) else None),
        validation_split=(float(validation_split) if validation_split is not None else None),
        output=str(Path(_string(config, "output")).resolve()),
        method=str(config.get("method", "lora")),
        epochs=float(config.get("epochs", 1)),
        batch_size=_positive_int(config.get("batch_size", 1), "batch_size"),
        gradient_accumulation_steps=_positive_int(
            config.get("gradient_accumulation_steps", 8),
            "gradient_accumulation_steps",
        ),
        max_sequence_length=_positive_int(
            config.get("max_sequence_length", 2048),
            "max_sequence_length",
        ),
        learning_rate=float(config.get("learning_rate", 0.0002)),
        precision=str(config.get("precision", "bf16")),
        seed=int(config.get("seed", 42)),
        max_steps=int(config.get("max_steps", -1)),
        logging_steps=_positive_int(config.get("logging_steps", 1), "logging_steps"),
        save_steps=_positive_int(config.get("save_steps", 50), "save_steps"),
        resume_from_checkpoint=bool(config.get("resume_from_checkpoint", True)),
        lora=LoraSettings(
            rank=_positive_int(lora_raw.get("rank", 16), "lora.rank"),
            alpha=_positive_int(lora_raw.get("alpha", 32), "lora.alpha"),
            dropout=float(lora_raw.get("dropout", 0.05)),
            target_modules=normalized_targets,
        ),
    )


def recipe_from_file(path: Path) -> TrainingRecipe:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Training request does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid training request JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("Training request must be a JSON object")
    return recipe_from_config(data)


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{key}' must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"'{name}' must be a positive integer")
    return value
