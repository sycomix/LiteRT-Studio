from __future__ import annotations

import json
from pathlib import Path

from litert_studio.cli import main


def test_cli_prints_conversion_plan(model_dir: Path, tmp_path: Path, capsys) -> None:
    config = tmp_path / "conversion.json"
    config.write_text(
        json.dumps({"source": str(model_dir), "output": str(tmp_path / "out")}),
        encoding="utf-8",
    )
    result = main(["plan-convert", str(config)])
    captured = capsys.readouterr()
    assert result == 0
    assert '"kind": "conversion"' in captured.out
    assert '"job_id":' in captured.out
