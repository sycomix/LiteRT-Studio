from __future__ import annotations

import sys
from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.process import SubprocessLauncher


def test_launcher_uses_argv_and_captures_output(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    handle = SubprocessLauncher().launch(
        [sys.executable, "-c", "print('worker ready')"],
        cwd=tmp_path,
        log_path=log,
    )
    assert handle.wait(timeout=10) == 0
    assert log.read_text(encoding="utf-8").strip() == "worker ready"


def test_launcher_rejects_unapproved_environment(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not allowed"):
        SubprocessLauncher().launch(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            log_path=tmp_path / "worker.log",
            environment={"SECRET_TOKEN": "must-not-be-passed"},
        )
