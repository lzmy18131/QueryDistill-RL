"""SQL execution environment keyed by ``db_id``.

The model policy only ever sees a ``db_id``. The environment resolves that
id through an explicit allowlist (registry file) into a filesystem path; the
model can never supply a database path itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from .executor import ExecutionResult, SafeSQLExecutor

_DB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class SQLExecutionEnvironment:
    """Resolve ``db_id -> allowlisted SQLite file`` and execute safely."""

    def __init__(
        self,
        db_paths: dict[str, Path],
        max_rows: int = 1000,
        max_execution_ms: int = 3000,
        worker_start_timeout: float = 30.0,
    ):
        sanitized: dict[str, Path] = {}
        for db_id, path in db_paths.items():
            self.validate_db_id(db_id)
            resolved = Path(path).resolve()
            if not resolved.is_file():
                raise ValueError(f"allowlisted database does not exist: {db_id} -> {resolved}")
            sanitized[db_id] = resolved
        self.db_paths = sanitized
        self.max_rows = max_rows
        self.max_execution_ms = max_execution_ms
        self.worker_start_timeout = worker_start_timeout

    @staticmethod
    def validate_db_id(db_id: str) -> str:
        if not isinstance(db_id, str) or not _DB_ID_RE.match(db_id):
            raise ValueError(
                f"invalid db_id {db_id!r}: must match {_DB_ID_RE.pattern} (no paths allowed)"
            )
        return db_id

    @classmethod
    def from_registry(
        cls,
        registry_path: str | Path,
        max_rows: int = 1000,
        max_execution_ms: int = 3000,
        worker_start_timeout: float = 30.0,
    ) -> SQLExecutionEnvironment:
        import json

        registry = Path(registry_path)
        payload = json.loads(registry.read_text(encoding="utf-8"))
        raw = payload.get("databases", payload)
        if not isinstance(raw, dict):
            raise ValueError(f"registry {registry} must contain a 'databases' mapping")
        db_paths: dict[str, Path] = {}
        for db_id, value in raw.items():
            cls.validate_db_id(db_id)
            path = Path(str(value))
            if not path.is_absolute():
                path = registry.parent / path
            resolved = path.resolve()
            try:
                resolved.relative_to(registry.parent.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"registry entry {db_id} escapes registry root: {resolved}"
                ) from exc
            db_paths[db_id] = resolved
        return cls(
            db_paths=db_paths,
            max_rows=max_rows,
            max_execution_ms=max_execution_ms,
            worker_start_timeout=worker_start_timeout,
        )

    def executor_for(self, db_id: str) -> SafeSQLExecutor:
        self.validate_db_id(db_id)
        if db_id not in self.db_paths:
            raise KeyError(f"db_id {db_id!r} is not in the allowlist")
        return SafeSQLExecutor(
            self.db_paths[db_id],
            max_rows=self.max_rows,
            max_execution_ms=self.max_execution_ms,
            worker_start_timeout=self.worker_start_timeout,
        )

    def execute(self, db_id: str, sql: str) -> ExecutionResult:
        return self.executor_for(db_id).execute(sql)

    def table_names(self, db_id: str) -> list[str]:
        result = self.execute(
            db_id, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        if not result.success:
            return []
        return [str(row[0]) for row in result.rows]

    def registry_snapshot(self) -> dict:
        return {
            "db_ids": sorted(self.db_paths),
            "databases": {db_id: str(path) for db_id, path in sorted(self.db_paths.items())},
            "max_rows": self.max_rows,
            "max_execution_ms": self.max_execution_ms,
        }
