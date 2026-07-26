from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from litert_studio.core.errors import ConfigurationError

SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    }
)


@dataclass
class ProcessHandle:
    process: subprocess.Popen[str]
    log: TextIO

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self.process.wait(timeout=timeout)
        finally:
            self.log.close()

    def cancel(self, timeout: float = 5) -> int:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        return int(self.process.returncode)


class SubprocessLauncher:
    """Launches argument arrays without a shell and with an allowlisted environment."""

    def launch(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessHandle:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ConfigurationError("Process argv must contain non-empty strings")
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise ConfigurationError(f"Process working directory does not exist: {cwd}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}
        for key, value in (environment or {}).items():
            if key not in SAFE_ENV_KEYS:
                log.close()
                raise ConfigurationError(f"Environment key is not allowed: {key}")
            env[key] = value
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
        except Exception:
            log.close()
            raise
        return ProcessHandle(process=process, log=log)
