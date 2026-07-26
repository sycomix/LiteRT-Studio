from __future__ import annotations

from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.training.datasets import inspect_jsonl


def test_records_after_preview_limit_are_still_validated(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"text": "preview"}\n{"unsupported": "must still fail"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Record 2"):
        inspect_jsonl(dataset, sample_limit=1)
