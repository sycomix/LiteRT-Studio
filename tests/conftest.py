from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "tiny", "architectures": ["TinyForCausalLM"]}),
        encoding="utf-8",
    )
    header = json.dumps(
        {"model.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"text": "hello"}\n'
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "content": "hello"}]}\n',
        encoding="utf-8",
    )
    return path
