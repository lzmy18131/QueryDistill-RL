"""Layer 2 of the SQL safety stack: process-isolated read-only SQLite executor.

Design decisions:

* Model SQL is executed in a **separate process** (spawn context), so a timeout
  can terminate the whole worker, not just a Python thread.
* The connection opens the database file in SQLite read-only mode
  (``file:...?mode=ro``) and installs an authorizer that only permits
  ``SELECT``/``READ``/``FUNCTION`` (and denies ``load_extension``). This is
  defense in depth on top of the sqlglot AST validation.
* A SQLite progress handler plus a watchdog ``interrupt()`` thread enforce
  ``max_execution_ms`` inside the worker; the parent process ``terminate()`` is
  the hard backstop.
* ``max_rows`` bounds how many rows can leave the worker.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .safety import validate_sql

# Authorizer action codes that are permitted for a read-only SELECT workload.
_ALLOWED_AUTHORIZER_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


@dataclass
class ExecutionResult:
    success: bool
    rows: list[list] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error_type: str = "none"
    error_message: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    row_count: int = 0
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "rows": self.rows,
            "columns": self.columns,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "row_count": self.row_count,
            "truncated": self.truncated,
        }


def _jsonable(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _authorizer(action: int, _arg1: object, arg2: object, _dbname: object, _source: object) -> int:
    if action in _ALLOWED_AUTHORIZER_ACTIONS:
        if action == sqlite3.SQLITE_FUNCTION and str(arg2 or "").lower() == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _run_sql_worker(
    db_path: str,
    sql: str,
    max_rows: int,
    max_execution_ms: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker entry point (top-level so the spawn context can import it)."""
    started = time.monotonic()
    connection: sqlite3.Connection | None = None
    watchdog: threading.Thread | None = None
    try:
        connection = sqlite3.connect(_read_only_uri(Path(db_path)), uri=True, timeout=5.0)
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(_authorizer)

        deadline = started + max_execution_ms / 1000.0

        def progress_check() -> int:
            if time.monotonic() > deadline:
                raise TimeoutError("SQL execution exceeded max_execution_ms")
            return 0

        connection.set_progress_handler(progress_check, 1000)

        def interrupt_after() -> None:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if connection is not None:
                with contextlib.suppress(Exception):  # connection may already be closed
                    connection.interrupt()

        watchdog = threading.Thread(target=interrupt_after, daemon=True)
        watchdog.start()

        cursor = connection.execute(sql)
        columns = [description[0] for description in (cursor.description or [])]
        rows: list[list] = []
        truncated = False
        while len(rows) < max_rows:
            if time.monotonic() > deadline:
                raise TimeoutError("SQL execution exceeded max_execution_ms while fetching")
            row = cursor.fetchone()
            if row is None:
                break
            rows.append([_jsonable(value) for value in row])
        if len(rows) >= max_rows:
            truncated = cursor.fetchone() is not None

        result_queue.put(
            {
                "success": True,
                "rows": rows,
                "columns": columns,
                "error_type": "none",
                "error_message": "",
                "duration_ms": round((time.monotonic() - started) * 1000.0, 2),
                "timed_out": False,
                "row_count": len(rows),
                "truncated": truncated,
            }
        )
    except TimeoutError as exc:
        result_queue.put(
            {
                "success": False,
                "rows": [],
                "columns": [],
                "error_type": "timeout",
                "error_message": str(exc),
                "duration_ms": round((time.monotonic() - started) * 1000.0, 2),
                "timed_out": True,
                "row_count": 0,
                "truncated": False,
            }
        )
    except sqlite3.Error as exc:
        interrupted = "interrupt" in str(exc).lower()
        result_queue.put(
            {
                "success": False,
                "rows": [],
                "columns": [],
                "error_type": "timeout" if interrupted else "sqlite_error",
                "error_message": str(exc),
                "duration_ms": round((time.monotonic() - started) * 1000.0, 2),
                "timed_out": interrupted,
                "row_count": 0,
                "truncated": False,
            }
        )
    except BaseException as exc:  # noqa: BLE001 - worker must always report back
        result_queue.put(
            {
                "success": False,
                "rows": [],
                "columns": [],
                "error_type": "internal_error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.monotonic() - started) * 1000.0, 2),
                "timed_out": False,
                "row_count": 0,
                "truncated": False,
            }
        )
    finally:
        try:
            if connection is not None:
                connection.close()
        except Exception:  # noqa: BLE001
            pass


def _failed_result(error_type: str, message: str) -> ExecutionResult:
    return ExecutionResult(
        success=False,
        error_type=error_type,
        error_message=message,
        timed_out=error_type == "timeout",
    )


def _process_context() -> multiprocessing.context.BaseContext:
    """Return the process context used for SQL worker isolation.

    On POSIX systems ``fork`` is dramatically faster and avoids the expensive
    re-import that makes ``spawn`` time out after CUDA has been initialized in
    the parent (observed in WSL).  The worker never touches CUDA, so forking a
    CUDA-initialized parent is safe for this narrowly-scoped worker.  On other
    platforms (notably Windows) we keep the portable ``spawn`` context.
    """
    if os.name == "posix":
        try:
            return multiprocessing.get_context("fork")
        except ValueError:
            pass
    return multiprocessing.get_context("spawn")


class SafeSQLExecutor:
    """Execute candidate SQL against one allowlisted SQLite file, read-only."""

    def __init__(
        self,
        db_path: str | Path,
        max_rows: int = 1000,
        max_execution_ms: int = 3000,
        worker_start_timeout: float = 30.0,
    ):
        self.db_path = Path(db_path)
        self.max_rows = int(max_rows)
        self.max_execution_ms = int(max_execution_ms)
        self.worker_start_timeout = float(worker_start_timeout)
        if self.max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if self.max_execution_ms <= 0:
            raise ValueError("max_execution_ms must be positive")
        if self.worker_start_timeout <= 0:
            raise ValueError("worker_start_timeout must be positive")

    def execute(self, sql: str) -> ExecutionResult:
        decision = validate_sql(sql)
        if not decision.safe:
            return _failed_result(decision.error_type, decision.reason)

        if not self.db_path.is_file():
            return _failed_result("schema_error", f"database file not found: {self.db_path.name}")

        context = _process_context()
        result_queue: multiprocessing.Queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_run_sql_worker,
            args=(str(self.db_path), sql, self.max_rows, self.max_execution_ms, result_queue),
            daemon=True,
        )
        started = time.monotonic()
        process.start()
        hard_deadline = started + self.max_execution_ms / 1000.0 + self.worker_start_timeout
        process.join(timeout=max(0.1, hard_deadline - time.monotonic()))

        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5.0)
            return ExecutionResult(
                success=False,
                error_type="timeout",
                error_message="worker process terminated after max_execution_ms hard deadline",
                duration_ms=round((time.monotonic() - started) * 1000.0, 2),
                timed_out=True,
            )

        try:
            payload = result_queue.get(timeout=2.0)
        except queue.Empty:
            return _failed_result(
                "internal_error", "worker exited without returning an execution result"
            )

        return ExecutionResult(
            success=bool(payload["success"]),
            rows=payload["rows"],
            columns=payload["columns"],
            error_type=payload["error_type"],
            error_message=payload["error_message"],
            duration_ms=payload["duration_ms"],
            timed_out=payload["timed_out"],
            row_count=payload["row_count"],
            truncated=payload["truncated"],
        )
