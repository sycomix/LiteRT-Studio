from __future__ import annotations

import json
from pathlib import Path

import pytest

from litert_studio.conversion.reference import (
    LogitValue,
    ReferenceCase,
    ReferenceSuite,
    load_prompts,
)
from litert_studio.core.errors import ConfigurationError


def test_reference_suite_does_not_contain_raw_prompt(tmp_path: Path) -> None:
    suite = ReferenceSuite(
        model_fingerprint="sha256-meta:model",
        adapter_sha256=None,
        top_k=1,
        max_new_tokens=1,
        cases=(
            ReferenceCase(
                prompt_sha256="a" * 64,
                input_ids=(1, 2),
                next_token_top_k=(LogitValue(3, 0.5),),
                generated_ids=(3,),
            ),
        ),
    )
    output = suite.write(tmp_path / "reference.json")
    content = output.read_text(encoding="utf-8")
    assert "raw secret prompt" not in content
    assert json.loads(content)["cases"][0]["input_ids"] == [1, 2]


def test_prompt_loader_validates_shape(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text('{"prompts": []}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="non-empty"):
        load_prompts(path)
