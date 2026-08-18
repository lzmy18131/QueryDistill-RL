"""Diagnostic-only fallback SQL extractor.

This is used ONLY for generation diagnostics and (if the diagnostic threshold
is met) for a controlled reward-parser extension. It never extracts multiple
statements, destructive SQL, or arbitrary prose with ambiguous queries.
"""

from __future__ import annotations

import re

import sqlglot

_SQL_START_RE = re.compile(r"(?is)\b(SELECT|WITH)\b")
_SQL_BLOCK_RE = re.compile(r"<\s*(?:sql)\s*>(.*?)<\s*/\s*(?:sql)\s*>", re.I | re.S)


def _is_single_select_like(statements: list) -> bool:
    from sqlglot import exp

    if len(statements) != 1:
        return False
    statement = statements[0]
    if statement is None or isinstance(statement, exp.Semicolon):
        return False
    return isinstance(statement, (exp.Select, exp.Union))


def _accept_single_select(sql: str) -> str | None:
    """Return ``sql`` if it parses as exactly one SELECT/WITH-SELECT."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
        statements = [
            s
            for s in statements
            if s is not None and not (hasattr(s, "key") and s.key == "semicolon")
        ]
        if _is_single_select_like(statements):
            return sql.strip()
    except Exception:
        return None
    return None


def _first_sql_block(text: str) -> str | None:
    """Return the single complete <sql>...</sql> block, if safe.

    If the output contains more than one complete SQL block, it is ambiguous
    and is rejected rather than scoring only the first block.  This prevents
    reward hacking where a model emits a correct SQL block followed by a
    destructive or arbitrary second block.
    """
    matches = _SQL_BLOCK_RE.findall(text)
    if len(matches) != 1:
        return None
    candidate = matches[0].strip()
    if not candidate:
        return None
    # Reject if the block itself contains multiple statements.
    semicolon = candidate.find(";")
    if semicolon != -1 and candidate[semicolon + 1 :].strip():
        return None
    return _accept_single_select(candidate)


def extract_fallback_sql(raw: str) -> str | None:
    """Extract exactly one read-only SELECT/WITH-SELECT from raw output.

    Returns None if the text is ambiguous, contains multiple statements, or is
    not a single select-like SQL expression.
    """
    if not raw:
        return None
    # First try the whole raw text.
    for candidate in (raw, raw.strip()):
        try:
            statements = sqlglot.parse(candidate, read="sqlite")
            statements = [
                s
                for s in statements
                if s is not None and not (hasattr(s, "key") and s.key == "semicolon")
            ]
            if _is_single_select_like(statements):
                return candidate.strip()
        except Exception:
            pass

    # Then try the first complete <sql>...</sql> block. This handles models
    # that emit the correct first block and then continue generating.
    block_sql = _first_sql_block(raw)
    if block_sql:
        return block_sql

    # Then try extracting the first SELECT/WITH ... segment.
    match = _SQL_START_RE.search(raw)
    if not match:
        return None
    start = match.start()
    candidate = raw[start:].strip()
    # A semicolon followed by more non-whitespace content indicates multiple
    # statements; reject rather than silently taking the first one.
    semicolon = candidate.find(";")
    if semicolon != -1 and candidate[semicolon + 1 :].strip():
        return None
    try:
        statements = sqlglot.parse(candidate, read="sqlite")
        statements = [
            s
            for s in statements
            if s is not None and not (hasattr(s, "key") and s.key == "semicolon")
        ]
        if _is_single_select_like(statements):
            return candidate.strip()
    except Exception:
        return None
    return None
