"""Aggregate artifact status files into one honest report."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings
from ..utils import atomic_write_json, atomic_write_text, load_json, utc_now_iso

_KNOWN_SMOKES = ("inference", "sft", "grpo", "gptq", "vllm")
_STATUS_ORDER = ("PASS", "FAIL", "BLOCKED", "SKIPPED", "DRY_RUN", "NOT_RUN")


def collect_smoke_statuses(settings: Settings | None = None) -> dict:
    settings = settings or Settings.load()
    smoke_root = settings.artifact_dir / "smoke"
    statuses: dict[str, dict] = {}
    for name in _KNOWN_SMOKES:
        status_path = smoke_root / name / "status.json"
        if status_path.exists():
            try:
                statuses[name] = load_json(status_path)
            except Exception as exc:  # noqa: BLE001
                statuses[name] = {"status": "FAIL", "reason": f"unreadable status.json: {exc}"}
        else:
            statuses[name] = {"status": "NOT_RUN"}
    return statuses


def build_report(settings: Settings | None = None, output_dir: str | Path | None = None) -> dict:
    settings = settings or Settings.load()
    output_dir = Path(output_dir or settings.project_root / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    statuses = collect_smoke_statuses(settings)
    report = {
        "generated_at": utc_now_iso(),
        "project_root": str(settings.project_root),
        "cache_root": str(settings.cache_root),
        "smoke_statuses": statuses,
        "first_round_rule": (
            "This round only performs smoke-level verification. PASS/FAIL/SKIPPED/BLOCKED "
            "are recorded per artifact; no benchmark claims are made."
        ),
    }
    atomic_write_json(output_dir / "status_report.json", report)

    lines = ["# QueryDistill-RL smoke status", "", f"generated_at: {report['generated_at']}", ""]
    for name, payload in statuses.items():
        status = payload.get("status", "NOT_RUN")
        lines.append(f"- {name}: **{status}**")
    atomic_write_text(output_dir / "STATUS.md", "\n".join(lines) + "\n")
    return report


def format_report(report: dict) -> str:
    lines = ["QueryDistill-RL smoke status"]
    for name, payload in report["smoke_statuses"].items():
        status = payload.get("status", "NOT_RUN")
        extra = payload.get("reason", "") or ""
        lines.append(f"  {name:<12} {status:<10} {extra}")
    return "\n".join(lines)
