from __future__ import annotations

from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.training.formatting import (
    load_formatted_records,
    split_formatted_records,
    token_statistics,
)


class FakeTokenizer:
    def __call__(self, text: str, **kwargs):
        return {"input_ids": text.split()}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return " ".join(f"{item['role']}: {item['content']}" for item in messages)


def test_formats_text_and_chat_records(dataset: Path) -> None:
    texts = load_formatted_records(dataset, FakeTokenizer())
    assert texts[0] == "hello"
    assert texts[1] == "user: hi assistant: hello"


def test_token_statistics_reports_truncation() -> None:
    stats = token_statistics(
        ["one two", "one two three four"],
        FakeTokenizer(),
        max_sequence_length=3,
    )
    assert stats.records == 2
    assert stats.minimum == 2
    assert stats.maximum == 4
    assert stats.mean == 3
    assert stats.truncated == 1


def test_validation_split_is_deterministic_and_disjoint() -> None:
    texts = [f"record-{index}" for index in range(10)]
    first = split_formatted_records(texts, validation_split=0.2, seed=17)
    second = split_formatted_records(texts, validation_split=0.2, seed=17)

    assert first == second
    assert len(first.train) == 8
    assert len(first.evaluation) == 2
    assert set(first.train).isdisjoint(first.evaluation)
    assert set(first.train + first.evaluation) == set(texts)


def test_validation_split_preserves_train_and_eval_for_small_dataset() -> None:
    split = split_formatted_records(["first", "second"], 0.9, seed=42)

    assert len(split.train) == 1
    assert len(split.evaluation) == 1


def test_validation_split_requires_two_records() -> None:
    with pytest.raises(ConfigurationError, match="At least two records"):
        split_formatted_records(["only"], 0.25, seed=42)
