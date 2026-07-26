from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from litert_studio.core.errors import ConfigurationError


class TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True)
class TokenStatistics:
    records: int
    minimum: int
    maximum: int
    mean: float
    truncated: int
    max_sequence_length: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[str, ...]
    evaluation: tuple[str, ...]
    seed: int
    validation_split: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "train_records": len(self.train),
            "evaluation_records": len(self.evaluation),
            "seed": self.seed,
            "validation_split": self.validation_split,
        }


def split_formatted_records(
    texts: list[str],
    validation_split: float,
    seed: int,
) -> DatasetSplit:
    if not 0 < validation_split < 1:
        raise ConfigurationError("'validation_split' must be between 0 and 1")
    if len(texts) < 2:
        raise ConfigurationError("At least two records are required for a validation split")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    evaluation_count = max(1, round(len(texts) * validation_split))
    evaluation_count = min(evaluation_count, len(texts) - 1)
    evaluation_indices = set(indices[:evaluation_count])
    return DatasetSplit(
        train=tuple(text for index, text in enumerate(texts) if index not in evaluation_indices),
        evaluation=tuple(text for index, text in enumerate(texts) if index in evaluation_indices),
        seed=seed,
        validation_split=validation_split,
    )


def load_formatted_records(path: Path, tokenizer: TokenizerLike) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ConfigurationError(f"Record {line_number} must be an object")
            if isinstance(record.get("text"), str) and record["text"]:
                texts.append(record["text"])
                continue
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ConfigurationError(
                    f"Record {line_number} must contain non-empty 'text' or 'messages'"
                )
            normalized: list[dict[str, str]] = []
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str)
                ):
                    raise ConfigurationError(
                        f"Record {line_number} contains an invalid chat message"
                    )
                normalized.append({"role": message["role"], "content": message["content"]})
            try:
                texts.append(
                    tokenizer.apply_chat_template(
                        normalized,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            except ValueError as exc:
                raise ConfigurationError(
                    f"Record {line_number} requires a tokenizer chat template: {exc}"
                ) from exc
    if not texts:
        raise ConfigurationError(f"Dataset contains no usable records: {path}")
    return texts


def token_statistics(
    texts: list[str],
    tokenizer: TokenizerLike,
    max_sequence_length: int,
) -> TokenStatistics:
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        token_ids = encoded.get("input_ids")
        if not isinstance(token_ids, list):
            raise ConfigurationError("Tokenizer did not return an input_ids list")
        lengths.append(len(token_ids))
    return TokenStatistics(
        records=len(lengths),
        minimum=min(lengths),
        maximum=max(lengths),
        mean=sum(lengths) / len(lengths),
        truncated=sum(length > max_sequence_length for length in lengths),
        max_sequence_length=max_sequence_length,
    )
