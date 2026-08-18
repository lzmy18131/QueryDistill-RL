"""BIRD official evaluator wrapper.

This module exports predictions in the official BIRD format and invokes the
downstream official evaluator scripts (third_party/bird_eval) without claiming
them as original work. It also computes an internal execution accuracy with the
project's own verifier so the two can be compared.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..data.schema import Example
from ..sql.environment import SQLExecutionEnvironment
from ..sql.verifier import ResultEquivalenceVerifier


def export_predictions(
    predictions: list[dict],
    output_path: str | Path,
) -> Path:
    """Write predictions in official BIRD JSONL format.

    Each prediction must contain ``example_id`` (or ``question_id``), ``db_id``,
    and ``SQL``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pred in predictions:
        question_id = pred.get("question_id")
        if question_id is None:
            example_id = str(pred.get("example_id", ""))
            if example_id.startswith("bird-dev-") and example_id[len("bird-dev-") :].isdigit():
                question_id = int(example_id[len("bird-dev-") :])
            else:
                question_id = example_id
        lines.append(
            json.dumps(
                {
                    "question_id": question_id,
                    "db_id": pred["db_id"],
                    "SQL": pred["SQL"],
                },
                ensure_ascii=False,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def internal_execution_accuracy(
    examples: list[Example],
    predictions: dict[str, str],
    environment: SQLExecutionEnvironment,
) -> dict:
    """Compute project-internal execution accuracy using strict equivalence."""
    verifier = ResultEquivalenceVerifier()
    correct = 0
    total = 0
    errors: list[dict] = []
    for example in examples:
        predicted_sql = predictions.get(example.example_id)
        if predicted_sql is None:
            errors.append({"example_id": example.example_id, "error": "missing_prediction"})
            continue
        total += 1
        try:
            candidate = environment.execute(example.db_id, predicted_sql)
            gold = environment.execute(example.db_id, example.gold_sql)
            verification = verifier.verify(
                candidate=candidate,
                gold=gold,
                candidate_sql=predicted_sql,
                gold_sql=example.gold_sql,
                schema_tables=set(environment.table_names(example.db_id)),
            )
            if verification.strict_equivalent:
                correct += 1
            else:
                errors.append(
                    {
                        "example_id": example.example_id,
                        "error": verification.kind,
                        "reason": verification.reason,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - evaluator reports any failure
            errors.append({"example_id": example.example_id, "error": str(exc)})
    return {
        "num_examples": total,
        "execution_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "errors": errors,
    }


def run_official_evaluator(
    predictions_path: str | Path,
    gold_path: str | Path,
    db_dir: str | Path,
    evaluator_script: str | Path = "third_party/bird_eval/evaluation_ex.py",
    timeout: int = 600,
) -> dict:
    """Run the official BIRD evaluator as an external process.

    Returns a dict with ``returncode``, ``stdout_tail``, and ``stderr_tail``.
    The caller is responsible for parsing the official accuracy if the script
    emits it; this wrapper never fabricates a number.
    """
    cmd = [
        "python",
        str(evaluator_script),
        "--db_dir",
        str(db_dir),
        "--gold",
        str(gold_path),
        "--predicted",
        str(predictions_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-50:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-50:]),
        }
    except Exception as exc:  # noqa: BLE001 - evaluator unavailable is recorded honestly
        return {"command": cmd, "error": str(exc)}
