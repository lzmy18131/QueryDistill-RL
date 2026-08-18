"""Generation utilities: stopping criteria and protocol-safe sampling helpers."""

from __future__ import annotations

from .stopping import SQL_CLOSE_TAG, StopAfterSqlClose

__all__ = ["SQL_CLOSE_TAG", "StopAfterSqlClose"]
