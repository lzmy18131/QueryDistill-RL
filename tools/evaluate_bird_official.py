#!/usr/bin/env python3
"""Run the official BIRD evaluator on an internal evaluation report.

Consumes the JSON report written by ``querydistill evaluate`` and produces the
official predicted.json + official_ex.log. This is bookkeeping only; the actual
evaluation is performed by third_party/bird_eval/evaluation_ex.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def extract_ex(stdout: str) -> dict:
    numbers = []
    for line in stdout.splitlines():
        match = re.match(r"^\s*EX\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if match:
            numbers = [float(x) for x in match.groups()]
    if len(numbers) != 4:
        return {}
    return {
        "official_simple_ex": numbers[0],
        "official_moderate_ex": numbers[1],
        "official_challenging_ex": numbers[2],
        "official_bird_execution_accuracy": numbers[3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--project-root",
        default="D:/LLMProjects/QueryDistill-RL",
    )
    args = parser.parse_args()

    root = Path(args.project_root)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted: dict[str, str] = {}
    missing = 0
    for record in report.get("records", []):
        example_id = record["example_id"]
        sql = record.get("sql")
        qid = example_id.rsplit("-", 1)[-1]
        if qid.isdigit():
            qid = str(int(qid))
        if not sql:
            # The official evaluator requires one prediction per question. A
            # missing/format-invalid prediction is honestly counted as wrong by
            # using a non-equivalent placeholder query; it is never counted as
            # correct.
            missing += 1
            sql = "SELECT 1"
        predicted[qid] = f"{sql}\t----- bird -----\t{record['db_id']}"

    predicted_path = output_dir / "predicted.json"
    predicted_path.write_text(json.dumps(predicted, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "third_party" / "bird_eval" / "evaluation_ex.py"),
        "--predicted_sql_path",
        str(predicted_path),
        "--ground_truth_path",
        str(root / "data" / "bird" / "eval" / "gold.txt"),
        "--db_root_path",
        str(root / "data" / "bird" / "eval" / "db_root") + "/",
        "--diff_json_path",
        str(root / "data" / "bird" / "eval" / "diff.jsonl"),
        "--output_log_path",
        str(output_dir / "official_ex.log"),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        stdout_tail = "\n".join(proc.stdout.splitlines()[-60:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-60:])
        parsed = extract_ex(proc.stdout)
        result = {
            "returncode": proc.returncode,
            "predicted_path": str(predicted_path),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "parsed": parsed,
            "missing_sql_predictions": missing,
        }
    except Exception as exc:  # noqa: BLE001
        result = {"error": str(exc), "missing_sql_predictions": missing}

    result_path = output_dir / "official_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
