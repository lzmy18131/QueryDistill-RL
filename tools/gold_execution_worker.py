#!/usr/bin/env python3
"""Standalone Gold SQL execution worker for the Phase 1.9 audit.

This worker is intentionally a separate script (not a multiprocessing child) so
the audit can run under Windows sandboxes where ``multiprocessing`` spawn handle
duplication can be denied.  It enforces the same read-only SQLite authorizer and
timeout semantics as SafeSQLExecutor's worker, and writes a JSON result file.

The parent process supplies the SQL through a file (or argv on small queries);
the worker never reads untrusted model SQL from stdin.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import threading
import time
from pathlib import Path

_ALLOWED_AUTHORIZER_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--sql-file", required=True)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--timeout-ms", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    connection: sqlite3.Connection | None = None
    watchdog: threading.Thread | None = None
    result: dict = {}
    try:
        connection = sqlite3.connect(_read_only_uri(Path(args.db_path)), uri=True, timeout=5.0)
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(_authorizer)

        sql = Path(args.sql_file).read_text(encoding="utf-8")
        max_rows = int(args.max_rows)
        max_execution_ms = int(args.timeout_ms)
        deadline = started + max_execution_ms / 1000.0

        def progress_check() -> int:
            if time.monotonic() > deadline:
                raise TimeoutError("SQL execution exceeded audit timeout")
            return 0

        connection.set_progress_handler(progress_check, 1000)

        def interrupt_after() -> None:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            if connection is not None:
                with contextlib.suppress(Exception):
                    connection.interrupt()

        watchdog = threading.Thread(target=interrupt_after, daemon=True)
        watchdog.start()

        cursor = connection.execute(sql)
        columns = [description[0] for description in (cursor.description or [])]
        rows: list[list] = []
        truncated = False
        while len(rows) < max_rows:
            if time.monotonic() > deadline:
                raise TimeoutError("SQL execution exceeded audit timeout while fetching")
            row = cursor.fetchone()
            if row is None:
                break
            rows.append([_jsonable(value) for value in row])
        if len(rows) >= max_rows:
            truncated = cursor.fetchone() is not None

        result = {
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
    except TimeoutError as exc:
        result = {
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
    except sqlite3.Error as exc:
        interrupted = "interrupt" in str(exc).lower()
        result = {
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
    except BaseException as exc:  # noqa: BLE001
        result = {
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
    finally:
        try:
            if connection is not None:
                connection.close()
        except Exception:  # noqa: BLE001
            pass

    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
