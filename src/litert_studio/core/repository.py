from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litert_studio.core.errors import ConfigurationError
from litert_studio.core.models import JobPlan, JobState

ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.PLANNED: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED}),
    JobState.RUNNING: frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    name: str
    state: JobState
    plan: dict[str, Any]
    created_at: str
    updated_at: str
    error: str | None


class SqliteJobRepository:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(self, plan: JobPlan) -> JobRecord:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(job_id, kind, name, state, plan_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.job_id,
                    plan.kind.value,
                    plan.name,
                    JobState.PLANNED.value,
                    json.dumps(plan.to_dict(), sort_keys=True),
                    now,
                    now,
                ),
            )
        return self.get(plan.job_id)

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, kind, name, state, plan_json, created_at, updated_at, error
                FROM jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise ConfigurationError(f"Unknown job: {job_id}")
        return JobRecord(
            job_id=row[0],
            kind=row[1],
            name=row[2],
            state=JobState(row[3]),
            plan=json.loads(row[4]),
            created_at=row[5],
            updated_at=row[6],
            error=row[7],
        )

    def list_recent(self, limit: int = 50) -> list[JobRecord]:
        if limit <= 0 or limit > 500:
            raise ConfigurationError("Job list limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, kind, name, state, plan_json, created_at, updated_at, error
                FROM jobs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            JobRecord(
                job_id=row[0],
                kind=row[1],
                name=row[2],
                state=JobState(row[3]),
                plan=json.loads(row[4]),
                created_at=row[5],
                updated_at=row[6],
                error=row[7],
            )
            for row in rows
        ]

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        error: str | None = None,
    ) -> JobRecord:
        current = self.get(job_id)
        if target not in ALLOWED_TRANSITIONS[current.state]:
            raise ConfigurationError(
                f"Invalid job transition: {current.state.value} -> {target.value}"
            )
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET state = ?, updated_at = ?, error = ?
                WHERE job_id = ? AND state = ?
                """,
                (target.value, _now(), error, job_id, current.state.value),
            ).rowcount
        if changed != 1:
            raise ConfigurationError(f"Job changed concurrently: {job_id}")
        return self.get(job_id)

    def add_event(
        self,
        job_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        self.get(job_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO job_events(job_id, created_at, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, _now(), event_type, json.dumps(payload or {}, sort_keys=True)),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an event identifier")
            return cursor.lastrowid

    def events(self, job_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        self.get(job_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, event_type, payload_json
                FROM job_events WHERE job_id = ? AND id > ? ORDER BY id
                """,
                (job_id, after_id),
            ).fetchall()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "event_type": row[2],
                "payload": json.loads(row[3]),
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_events_job_id
                ON job_events(job_id, id);
                """
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
