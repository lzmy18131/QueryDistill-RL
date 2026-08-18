"""Atomic file I/O and small shared utilities.

All checkpoints, datasets and progress files are written through atomic
replace so an interrupted process never leaves a half-written file.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

_JSONL_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class UnknownConfigFieldError(ValueError):
    """Raised when a YAML/JSON config contains a field the target model does not know."""


def strict_dataclass_from_dict(cls: type, payload: dict, source: str = "<config>"):
    """Construct a dataclass from a dict, rejecting unknown fields.

    This is the fail-closed replacement for ``{k: v for k, v in payload.items()
    if k in allowed}``. A typo such as ``model_id`` in a GRPO config must raise
    instead of being silently ignored.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise UnknownConfigFieldError(
            f"{source}: unknown config field(s) for {cls.__name__}: {unknown}"
        )
    return cls(**payload)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically via a temp file + ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path, payload: object, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, ensure_ascii=False) + "\n")


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _JSONL_LOCKS.setdefault(key, threading.Lock())


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Append JSON lines under a per-file lock; each line is one record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path), path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return records


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:  # noqa: BLE001 - reporting is best effort
        return None


def ensure_within(root: Path, path: Path) -> bool:
    """True when ``path`` resolves inside ``root`` (path traversal guard)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
