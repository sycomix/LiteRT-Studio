from __future__ import annotations

from typing import Protocol

from litert_studio.core.models import JobPlan, JobState, Stage
from litert_studio.core.repository import SqliteJobRepository


class StageExecutor(Protocol):
    def execute(self, stage: Stage, plan: JobPlan) -> None: ...


class LocalJobRunner:
    """Runs stage executors synchronously with durable lifecycle events."""

    def __init__(self, repository: SqliteJobRepository, executor: StageExecutor) -> None:
        self.repository = repository
        self.executor = executor

    def run(self, plan: JobPlan) -> JobState:
        try:
            self.repository.create(plan)
            self.repository.transition(plan.job_id, JobState.QUEUED)
            self.repository.add_event(plan.job_id, "job.queued")
            self.repository.transition(plan.job_id, JobState.RUNNING)
            self.repository.add_event(plan.job_id, "job.started")
            for index, stage in enumerate(plan.stages):
                self.repository.add_event(
                    plan.job_id,
                    "stage.started",
                    {"index": index, "name": stage.name, "backend": stage.backend},
                )
                self.executor.execute(stage, plan)
                self.repository.add_event(
                    plan.job_id,
                    "stage.succeeded",
                    {"index": index, "name": stage.name},
                )
            self.repository.transition(plan.job_id, JobState.SUCCEEDED)
            self.repository.add_event(plan.job_id, "job.succeeded")
            return JobState.SUCCEEDED
        except Exception as exc:
            record = self.repository.get(plan.job_id)
            if record.state is JobState.RUNNING:
                self.repository.transition(
                    plan.job_id,
                    JobState.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.repository.add_event(
                    plan.job_id,
                    "job.failed",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                )
            return JobState.FAILED
