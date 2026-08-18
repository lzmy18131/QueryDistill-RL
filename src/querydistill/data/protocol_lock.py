"""Validation helpers for the formal protocol lock.

Every hash in the lock must be a lowercase SHA-256 hex string (64 characters).
Placeholders such as ``[]``, ``null``, ``""`` and ``TBD`` are invalid.
"""

from __future__ import annotations

import re
from typing import Any

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def is_valid_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


def is_valid_git_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(_GIT_COMMIT_RE.match(value))


def protocol_hash_fields(lock: dict) -> list[tuple[str, Any]]:
    """Flatten all hash-bearing fields in a protocol lock.

    Fields are grouped by section.  The function returns ``(path, value)``
    pairs so callers can produce precise error messages.
    """
    fields: list[tuple[str, Any]] = []

    def add(path: str, value: Any) -> None:
        if path.endswith("_sha256") or path.endswith("_hash") or path == "git_commit":
            fields.append((path, value))

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    walk(value, path)
                elif isinstance(value, list):
                    # Nested lists (e.g. hash_conflicts) are not hash fields.
                    continue
                else:
                    add(path, value)
        else:
            add(prefix, obj)

    walk(lock)
    return fields


def validate_protocol_lock_hash_fields(lock: dict) -> list[str]:
    """Return a list of invalid hash-field descriptions (empty means valid).

    ``git_commit`` is a Git commit id (40 hex chars) rather than a SHA-256
    content hash, so it is validated with its own pattern.
    """
    errors: list[str] = []
    for path, value in protocol_hash_fields(lock):
        if path == "git_commit":
            if not is_valid_git_commit(value):
                errors.append(f"{path}={value!r} is not a 40-char lowercase Git commit id")
        elif not is_valid_sha256_hex(value):
            errors.append(f"{path}={value!r} is not a 64-char lowercase SHA-256 hex")
    return errors
