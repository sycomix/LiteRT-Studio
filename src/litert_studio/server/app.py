from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from litert_studio.conversion import build_conversion_plan
from litert_studio.conversion.adapters import AdapterRegistry
from litert_studio.conversion.benchmark import _quantization_for_model, benchmark_litert
from litert_studio.conversion.export_request import export_request_from_config
from litert_studio.conversion.inspector import inspect_model_directory
from litert_studio.conversion.litert_runtime import validate_litert_tokens
from litert_studio.conversion.quantization import quantization_policies
from litert_studio.conversion.reference import capture_reference_suite
from litert_studio.conversion.tensors import inspect_export_compatibility
from litert_studio.core.compatibility import CompatibilityRegistry
from litert_studio.core.devices import discover_android_devices, install_android_apk
from litert_studio.core.errors import StudioError
from litert_studio.core.model_import import import_huggingface_model
from litert_studio.core.models import JobState
from litert_studio.core.packaging import create_bundle, verify_bundle
from litert_studio.core.process import ProcessHandle, SubprocessLauncher
from litert_studio.core.repository import SqliteJobRepository
from litert_studio.training import build_training_plan
from litert_studio.training.datasets import inspect_jsonl
from litert_studio.training.merge import merge_adapter
from litert_studio.training.recipe import recipe_from_config


def create_app(workspace: Path | None = None):  # type: ignore[no-untyped-def]
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Install LiteRT Studio with the 'api' extra") from exc

    root = (workspace or Path.cwd()).resolve()
    state = root / ".litert-studio"
    state.mkdir(parents=True, exist_ok=True)
    repository = SqliteJobRepository(state / "jobs.sqlite3")
    compatibility = CompatibilityRegistry(state / "compatibility.sqlite3")
    processes: dict[str, ProcessHandle] = {}
    process_lock = Lock()
    launcher = SubprocessLauncher()
    static = Path(__file__).with_name("static")
    app = FastAPI(title="LiteRT Studio", version="0.1.0")

    def authorized(value: str, *, must_exist: bool = False) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root):
            raise HTTPException(403, "Path is outside the Studio workspace")
        if must_exist and not path.exists():
            raise HTTPException(404, f"Path does not exist: {path}")
        return path

    @app.exception_handler(StudioError)
    async def studio_error(_request: Any, exc: StudioError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "workspace": str(root)}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "conversion": [item.to_dict() for item in AdapterRegistry().list_capabilities()],
            "training_methods": ["lora", "qlora", "full"],
            "quantization": [item.to_dict() for item in quantization_policies()],
        }

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        packages = {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "litert-torch",
                "litert-lm",
            )
        }
        accelerator: dict[str, Any] = {"cuda": False, "devices": []}
        try:
            import torch

            accelerator = {
                "cuda": torch.cuda.is_available(),
                "devices": [
                    {
                        "name": torch.cuda.get_device_name(index),
                        "memory_gib": round(
                            torch.cuda.get_device_properties(index).total_memory / (1024**3),
                            2,
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
        except ImportError:
            pass
        return {
            "platform": platform.system().lower(),
            "release": platform.release(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "accelerator": accelerator,
            "packages": packages,
        }

    @app.get("/api/devices/android")
    def android_devices() -> dict[str, Any]:
        return discover_android_devices().to_dict()

    @app.post("/api/actions/install-android-app")
    def install_android_action(payload: dict[str, Any]) -> dict[str, Any]:
        serial = payload.get("serial")
        return install_android_apk(
            authorized(str(payload.get("apk", "")), must_exist=True),
            str(serial) if serial else None,
        )

    @app.get("/api/compatibility")
    def compatibility_results(limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0 or limit > 1000:
            raise HTTPException(422, "Limit must be between 1 and 1,000")
        return [item.to_dict() for item in compatibility.list_recent(limit)]

    @app.post("/api/models/inspect")
    def inspect_model(payload: dict[str, Any]) -> dict[str, Any]:
        model = authorized(str(payload.get("path", "")), must_exist=True)
        inspection = inspect_model_directory(model)
        compatibility = inspect_export_compatibility(model)
        return {
            "inspection": inspection.to_dict(),
            "export": compatibility.to_dict(),
        }

    @app.post("/api/models/import")
    def import_model(payload: dict[str, Any]) -> dict[str, Any]:
        imported = import_huggingface_model(
            str(payload.get("repository", "")),
            str(payload.get("revision", "")),
            root / "models",
        )
        return {
            "repository": imported.repository,
            "requested_revision": imported.requested_revision,
            "resolved_revision": imported.resolved_revision,
            "path": str(imported.path),
            "inspection": imported.inspection.to_dict(),
        }

    @app.post("/api/datasets/inspect")
    def inspect_dataset(payload: dict[str, Any]) -> dict[str, Any]:
        dataset = authorized(str(payload.get("path", "")), must_exist=True)
        return inspect_jsonl(dataset).to_dict()

    @app.post("/api/plans/training")
    def training_plan(payload: dict[str, Any]) -> dict[str, Any]:
        config = _authorize_config_paths(
            payload,
            authorized,
            ("base_model", "dataset", "eval_dataset", "output"),
        )
        plan = build_training_plan(config)
        return plan.to_dict()

    @app.post("/api/plans/conversion")
    def conversion_plan(payload: dict[str, Any]) -> dict[str, Any]:
        config = _authorize_config_paths(
            payload,
            authorized,
            ("source", "representative_dataset", "output"),
        )
        plan = build_conversion_plan(config)
        return plan.to_dict()

    @app.post("/api/jobs")
    def save_job(payload: dict[str, Any]) -> dict[str, Any]:
        kind = str(payload.get("kind", ""))
        config = payload.get("config")
        if not isinstance(config, dict):
            raise HTTPException(422, "'config' must be an object")
        if kind == "training":
            normalized = _authorize_config_paths(
                config, authorized, ("base_model", "dataset", "eval_dataset", "output")
            )
            plan = build_training_plan(normalized)
        elif kind == "conversion":
            normalized = _authorize_config_paths(
                config, authorized, ("source", "representative_dataset", "output")
            )
            plan = build_conversion_plan(normalized)
        else:
            raise HTTPException(422, "'kind' must be training or conversion")
        try:
            record = repository.create(plan)
        except Exception:
            record = repository.get(plan.job_id)
        return _record(record)

    def start_process_job(plan: Any, argv: list[str]) -> dict[str, Any]:
        try:
            record = repository.create(plan)
        except Exception:
            record = repository.get(plan.job_id)
        if record.state is not JobState.PLANNED:
            raise HTTPException(409, f"Job is already {record.state.value}")
        repository.transition(plan.job_id, JobState.QUEUED)
        repository.add_event(plan.job_id, "job.queued")
        repository.transition(plan.job_id, JobState.RUNNING)
        log_path = state / "logs" / f"{plan.job_id}.log"
        try:
            handle = launcher.launch(
                argv,
                cwd=root,
                log_path=log_path,
                environment={"PYTHONPATH": str(root / "src")},
            )
        except Exception as exc:
            repository.transition(
                plan.job_id,
                JobState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        with process_lock:
            processes[plan.job_id] = handle
        repository.add_event(
            plan.job_id,
            "job.started",
            {"pid": handle.pid, "log": str(log_path)},
        )

        def watch() -> None:
            status = handle.wait()
            with process_lock:
                processes.pop(plan.job_id, None)
            current = repository.get(plan.job_id)
            if current.state is JobState.CANCELLED:
                repository.add_event(
                    plan.job_id,
                    "job.process-ended",
                    {"exit_code": status},
                )
                return
            if status == 0:
                repository.transition(plan.job_id, JobState.SUCCEEDED)
                repository.add_event(
                    plan.job_id,
                    "job.succeeded",
                    {"exit_code": status},
                )
            else:
                repository.transition(
                    plan.job_id,
                    JobState.FAILED,
                    error=f"Worker exited with status {status}",
                )
                repository.add_event(
                    plan.job_id,
                    "job.failed",
                    {"exit_code": status},
                )

        Thread(target=watch, daemon=True, name=f"studio-{plan.job_id}").start()
        return _record(repository.get(plan.job_id))

    @app.post("/api/run/training")
    def run_training(payload: dict[str, Any]) -> dict[str, Any]:
        config = _authorize_config_paths(
            payload,
            authorized,
            ("base_model", "dataset", "eval_dataset", "output"),
        )
        plan = build_training_plan(config)
        recipe = recipe_from_config(config)
        request_path = state / "requests" / f"training-{recipe.request_id}.json"
        recipe.write(request_path)
        return start_process_job(
            plan,
            [
                sys.executable,
                "-m",
                "litert_studio.training.worker",
                "--request",
                str(request_path),
                "--execute",
            ],
        )

    @app.post("/api/run/conversion")
    def run_conversion(payload: dict[str, Any]) -> dict[str, Any]:
        config = _authorize_config_paths(
            payload,
            authorized,
            ("source", "representative_dataset", "output"),
        )
        plan = build_conversion_plan(config)
        request = export_request_from_config(config)
        request_path = state / "requests" / f"export-{request.request_id}.json"
        request.write(request_path)
        return start_process_job(
            plan,
            [
                sys.executable,
                "-m",
                "litert_studio.conversion.export_worker",
                "--request",
                str(request_path),
                "--execute",
            ],
        )

    @app.post("/api/actions/merge-adapter")
    def merge_action(payload: dict[str, Any]) -> dict[str, Any]:
        result = merge_adapter(
            authorized(str(payload.get("base_model", "")), must_exist=True),
            authorized(str(payload.get("adapter", "")), must_exist=True),
            authorized(str(payload.get("output", ""))),
        )
        return {
            "model_directory": str(result.model_directory),
            "manifest": str(result.manifest),
            "reused": result.reused,
        }

    @app.post("/api/actions/capture-reference")
    def reference_action(payload: dict[str, Any]) -> dict[str, Any]:
        adapter_value = payload.get("adapter")
        suite = capture_reference_suite(
            authorized(str(payload.get("model", "")), must_exist=True),
            authorized(str(payload.get("prompts", "")), must_exist=True),
            authorized(str(payload.get("output", ""))),
            adapter_dir=(
                authorized(str(adapter_value), must_exist=True)
                if isinstance(adapter_value, str) and adapter_value
                else None
            ),
            top_k=int(payload.get("top_k", 8)),
            max_new_tokens=int(payload.get("max_new_tokens", 8)),
        )
        return {"cases": len(suite.cases), "output": str(payload.get("output", ""))}

    @app.post("/api/actions/validate-litert")
    def validate_action(payload: dict[str, Any]) -> dict[str, Any]:
        model = authorized(str(payload.get("model", "")), must_exist=True)
        output = authorized(str(payload.get("output", "")))
        report = validate_litert_tokens(
            model,
            authorized(str(payload.get("reference", "")), must_exist=True),
            authorized(str(payload.get("prompts", "")), must_exist=True),
            output,
        )
        compatibility.record(
            model_sha256=report.model_sha256,
            quantization=_quantization_for_model(model),
            runtime=f"litert-lm {report.litert_lm_version}",
            device=report.backend,
            result_type="token_parity",
            passed=report.passed,
            report_path=output,
            summary={"cases": len(report.cases)},
        )
        return report.to_dict()

    @app.post("/api/actions/benchmark-litert")
    def benchmark_action(payload: dict[str, Any]) -> dict[str, Any]:
        model = authorized(str(payload.get("model", "")), must_exist=True)
        output = authorized(str(payload.get("output", "")))
        report = benchmark_litert(
            model,
            authorized(str(payload.get("prompts", "")), must_exist=True),
            output,
            warmup_iterations=int(payload.get("warmup_iterations", 1)),
            measured_iterations=int(payload.get("measured_iterations", 3)),
            max_output_tokens=int(payload.get("max_output_tokens", 16)),
        )
        compatibility.record(
            model_sha256=report.model_sha256,
            quantization=report.quantization,
            runtime=f"litert-lm {report.litert_lm_version}",
            device=report.device,
            result_type="benchmark",
            passed=True,
            report_path=output,
            summary=report.summary,
        )
        return report.to_dict()

    @app.post("/api/actions/package")
    def package_action(payload: dict[str, Any]) -> dict[str, Any]:
        reports = payload.get("reports", [])
        if not isinstance(reports, list):
            raise HTTPException(422, "'reports' must be a list")
        result = create_bundle(
            authorized(str(payload.get("artifact_directory", "")), must_exist=True),
            authorized(str(payload.get("output", ""))),
            reports=tuple(authorized(str(item), must_exist=True) for item in reports),
        )
        return asdict(result)

    @app.post("/api/actions/verify-bundle")
    def verify_action(payload: dict[str, Any]) -> dict[str, Any]:
        return asdict(verify_bundle(authorized(str(payload.get("bundle", "")), must_exist=True)))

    @app.get("/api/jobs")
    def jobs(limit: int = 50) -> list[dict[str, Any]]:
        return [_record(record) for record in repository.list_recent(limit)]

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return repository.events(job_id, after_id)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, limit: int = 100_000) -> dict[str, Any]:
        repository.get(job_id)
        if limit <= 0 or limit > 1_000_000:
            raise HTTPException(422, "Log limit must be between 1 and 1,000,000")
        path = state / "logs" / f"{job_id}.log"
        return {"job_id": job_id, "log": _tail(path, limit)}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        record = repository.get(job_id)
        if record.state not in {JobState.QUEUED, JobState.RUNNING}:
            raise HTTPException(409, f"Job is already {record.state.value}")
        repository.transition(job_id, JobState.CANCELLED)
        repository.add_event(job_id, "job.cancel-requested")
        with process_lock:
            handle = processes.get(job_id)
        exit_code = handle.cancel() if handle is not None else None
        repository.add_event(job_id, "job.cancelled", {"exit_code": exit_code})
        return _record(repository.get(job_id))

    app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/")
    def index():  # type: ignore[no-untyped-def]
        return FileResponse(static / "index.html")

    return app


def _authorize_config_paths(
    payload: dict[str, Any],
    authorize: Any,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    config = dict(payload)
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value:
            config[key] = str(authorize(value, must_exist=key not in {"output"}))
    return config


def _record(record: Any) -> dict[str, Any]:
    value = asdict(record)
    value["state"] = record.state.value
    return value


def _tail(path: Path, limit: int) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
