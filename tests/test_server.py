from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any

from fastapi.testclient import TestClient

from litert_studio.server.app import create_app


def test_studio_serves_gui_and_health(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    page = client.get("/")
    assert page.status_code == 200
    assert "LiteRT Studio" in page.text
    assert client.get("/health").json()["status"] == "ok"


def test_studio_inspects_workspace_model(model_dir: Path, tmp_path: Path) -> None:
    destination = tmp_path / "models" / "tiny"
    destination.parent.mkdir()
    model_dir.rename(destination)
    client = TestClient(create_app(tmp_path))

    response = client.post("/api/models/inspect", json={"path": "models/tiny"})

    assert response.status_code == 200
    assert response.json()["inspection"]["model_type"] == "tiny"
    assert response.json()["export"]["compatible"] is False


def test_studio_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))

    response = client.post("/api/models/inspect", json={"path": str(tmp_path.parent)})

    assert response.status_code == 403


def test_studio_reports_system_capabilities(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).get("/api/system")

    assert response.status_code == 200
    assert response.json()["python"]
    assert "accelerator" in response.json()


def test_training_job_runs_in_logged_subprocess(
    model_dir: Path,
    dataset: Path,
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/run/training",
        json={
            "base_model": str(model_dir),
            "dataset": str(dataset),
            "output": str(tmp_path / "output"),
            "method": "lora",
            "precision": "fp32",
            "max_steps": 1,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    state = "running"
    for _ in range(150):
        jobs = client.get("/api/jobs").json()
        state = next(job["state"] for job in jobs if job["job_id"] == job_id)
        if state in {"succeeded", "failed"}:
            break
        time.sleep(0.1)

    assert state in {"succeeded", "failed"}
    log = client.get(f"/api/jobs/{job_id}/log")
    assert log.status_code == 200
    assert isinstance(log.json()["log"], str)


def test_active_job_can_be_cancelled(
    model_dir: Path,
    dataset: Path,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    stopped = Event()

    class FakeHandle:
        pid = 123

        def wait(self) -> int:
            stopped.wait(5)
            return -15

        def cancel(self) -> int:
            stopped.set()
            return -15

    monkeypatch.setattr(
        "litert_studio.server.app.SubprocessLauncher.launch",
        lambda *_args, **_kwargs: FakeHandle(),
    )
    client = TestClient(create_app(tmp_path))
    started = client.post(
        "/api/run/training",
        json={
            "base_model": str(model_dir),
            "dataset": str(dataset),
            "output": str(tmp_path / "output"),
            "method": "lora",
            "precision": "fp32",
        },
    )
    job_id = started.json()["job_id"]

    cancelled = client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert stopped.is_set()
