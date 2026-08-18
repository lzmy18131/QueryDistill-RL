"""Data audit: schema checks, fixture DB checks, leakage checks, gold sanity."""

from __future__ import annotations

import json
from pathlib import Path

from ..sql.environment import SQLExecutionEnvironment
from .leakage import LeakageGuard
from .schema import load_distillation_records, load_examples


def audit_data(
    examples_path: str | Path,
    registry_path: str | Path,
    distillation_path: str | Path | None = None,
) -> dict:
    examples_path = Path(examples_path)
    registry_path = Path(registry_path)
    examples = load_examples(examples_path)
    environment = SQLExecutionEnvironment.from_registry(registry_path)
    guard = LeakageGuard()

    reports = {
        "examples": {"path": str(examples_path), "count": len(examples)},
        "split_counts": {},
        "duplicate_example_ids": False,
        "db_ids_missing_from_registry": [],
        "registry": json.loads(registry_path.read_text(encoding="utf-8")),
        "gold_execution": {"total": 0, "success": 0, "failed": []},
        "leakage": {},
        "distillation": None,
    }
    for example in examples:
        reports["split_counts"][example.split] = reports["split_counts"].get(example.split, 0) + 1
        if example.db_id not in environment.db_paths:
            reports["db_ids_missing_from_registry"].append(example.example_id)

    leakage_report = guard.audit_examples(examples)
    reports["leakage"] = leakage_report.as_dict()

    for example in examples:
        result = environment.execute(example.db_id, example.gold_sql)
        reports["gold_execution"]["total"] += 1
        if result.success:
            reports["gold_execution"]["success"] += 1
        else:
            reports["gold_execution"]["failed"].append(
                {
                    "example_id": example.example_id,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                }
            )

    if distillation_path:
        distillation_path = Path(distillation_path)
        if distillation_path.exists():
            records = load_distillation_records(distillation_path)
            candidate_ids = [record.example_id for record in records]
            distillation_leakage = guard.audit_examples(
                examples, candidate_example_ids=candidate_ids
            )
            reports["distillation"] = {
                "path": str(distillation_path),
                "record_count": len(records),
                "verified_count": sum(
                    1
                    for record in records
                    if record.parse_valid
                    and record.safe
                    and record.execution_success
                    and record.execution_equivalent
                ),
                "leakage": distillation_leakage.as_dict(),
            }
        else:
            reports["distillation"] = {
                "path": str(distillation_path),
                "error": "distillation file not found",
                "leakage": {"clean": False, "violations": []},
            }

    distillation_ok = reports["distillation"] is None or bool(
        reports["distillation"].get("leakage", {}).get("clean", False)
    )
    reports["ok"] = (
        leakage_report.clean
        and reports["gold_execution"]["success"] == reports["gold_execution"]["total"]
        and not reports["db_ids_missing_from_registry"]
        and distillation_ok
    )
    return reports


def format_audit(report: dict) -> str:
    lines = [
        f"data audit ok: {report['ok']}",
        f"examples: {report['examples']['count']} ({report['split_counts']})",
        f"gold execution: {report['gold_execution']['success']}/{report['gold_execution']['total']}",
        f"leakage clean: {report['leakage']['clean']}",
    ]
    for violation in report["leakage"]["violations"]:
        lines.append(f"  LEAKAGE [{violation['rule_id']}] {violation['detail']}")
    if report["distillation"]:
        lines.append(
            f"distillation: {report['distillation']['record_count']} records, "
            f"{report['distillation']['verified_count']} verified"
        )
    return "\n".join(lines)
