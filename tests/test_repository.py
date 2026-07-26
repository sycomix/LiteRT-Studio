from __future__ import annotations

from pathlib import Path

import pytest

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.models import JobKind, JobPlan, JobState, Stage
from litert_studio.core.repository import SqliteJobRepository
from litert_studio.core.runner import LocalJobRunner


def _plan(name: str = "test") -> JobPlan:
    return JobPlan(
        kind=JobKind.CONVERSION,
        name=name,
        stages=(
            Stage("one", "First stage", "test"),
            Stage("two", "Second stage", "test"),
        ),
        inputs={},
        outputs={},
    )


class RecordingExecutor:
    def __init__(self, fail_on: str | None = None) -> None:
        self.names: list[str] = []
        self.fail_on = fail_on

    def execute(self, stage: Stage, plan: JobPlan) -> None:
        self.names.append(stage.name)
        if stage.name == self.fail_on:
            raise RuntimeError("planned failure")


def test_repository_enforces_state_machine(tmp_path: Path) -> None:
    repository = SqliteJobRepository(tmp_path / "jobs.sqlite3")
    plan = _plan()
    repository.create(plan)
    with pytest.raises(ConfigurationError, match="Invalid job transition"):
        repository.transition(plan.job_id, JobState.SUCCEEDED)


def test_runner_persists_success_events(tmp_path: Path) -> None:
    repository = SqliteJobRepository(tmp_path / "jobs.sqlite3")
    executor = RecordingExecutor()
    state = LocalJobRunner(repository, executor).run(_plan())
    assert state is JobState.SUCCEEDED
    assert executor.names == ["one", "two"]
    assert repository.get(_plan().job_id).state is JobState.SUCCEEDED
    assert repository.events(_plan().job_id)[-1]["event_type"] == "job.succeeded"


def test_runner_records_failure_without_continuing(tmp_path: Path) -> None:
    repository = SqliteJobRepository(tmp_path / "jobs.sqlite3")
    executor = RecordingExecutor(fail_on="one")
    plan = _plan("failure")
    state = LocalJobRunner(repository, executor).run(plan)
    record = repository.get(plan.job_id)
    assert state is JobState.FAILED
    assert record.state is JobState.FAILED
    assert "planned failure" in (record.error or "")
    assert executor.names == ["one"]
