"""Status report aggregation for artifacts/smoke."""

from .report import build_report, collect_smoke_statuses

__all__ = ["build_report", "collect_smoke_statuses"]
