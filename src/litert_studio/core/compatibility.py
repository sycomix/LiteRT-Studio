from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class CompatibilityResult:
    result_id: str
    created_at: str
    model_sha256: str
    quantization: str
    runtime: str
    device: str
    result_type: str
    passed: bool
    report_path: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompatibilityRegistry:
    def __init__(self, database: Path) -> None:
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS compatibility_results (
                    result_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    quantization TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    device TEXT NOT NULL,
                    result_type TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    report_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        *,
        model_sha256: str,
        quantization: str,
        runtime: str,
        device: str,
        result_type: str,
        passed: bool,
        report_path: Path,
        summary: dict[str, Any],
    ) -> CompatibilityResult:
        result = CompatibilityResult(
            result_id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_sha256=model_sha256,
            quantization=quantization,
            runtime=runtime,
            device=device,
            result_type=result_type,
            passed=passed,
            report_path=str(report_path.resolve()),
            summary=summary,
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO compatibility_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.result_id,
                    result.created_at,
                    result.model_sha256,
                    result.quantization,
                    result.runtime,
                    result.device,
                    result.result_type,
                    int(result.passed),
                    result.report_path,
                    json.dumps(result.summary, sort_keys=True),
                ),
            )
        return result

    def list_recent(self, limit: int = 100) -> tuple[CompatibilityResult, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT result_id, created_at, model_sha256, quantization, runtime,
                       device, result_type, passed, report_path, summary_json
                FROM compatibility_results ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(
            CompatibilityResult(
                result_id=row[0],
                created_at=row[1],
                model_sha256=row[2],
                quantization=row[3],
                runtime=row[4],
                device=row[5],
                result_type=row[6],
                passed=bool(row[7]),
                report_path=row[8],
                summary=json.loads(row[9]),
            )
            for row in rows
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)
