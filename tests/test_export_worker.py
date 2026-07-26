from __future__ import annotations

from pathlib import Path

from test_gemma_mapping import _gemma_dir

import litert_studio.conversion.export_worker as worker
from litert_studio.conversion.export_request import LiteRTExportRequest


def test_windows_preflight_reports_linux_requirement(tmp_path: Path, monkeypatch, capsys) -> None:
    model = _gemma_dir(tmp_path)
    request = LiteRTExportRequest(
        model=str(model),
        output_dir=str(tmp_path / "output"),
    )
    request_path = request.write(tmp_path / "request.json")
    monkeypatch.setattr(worker.platform, "system", lambda: "Windows")
    result = worker.main(["--request", str(request_path)])
    assert result == 2
    assert "requires Linux" in capsys.readouterr().out
