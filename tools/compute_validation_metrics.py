#!/usr/bin/env python3
"""Compute validation metrics from run_validation_diagnostic output JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute(path: str | Path) -> dict:
    recs = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(recs)

    def rate(key: str) -> float | None:
        return round(sum(1 for r in recs if r.get(key)) / n, 4) if n else None

    metrics = {
        "records": n,
        "format_valid_rate": rate("format_valid"),
        "parse_valid_rate": rate("sql_parse_valid"),
        "safe_rate": rate("safe"),
        "execution_success_rate": rate("execution_success"),
        "strict_execution_accuracy": rate("strict_equivalent"),
        "stop_at_sql_close_rate": round(
            sum(1 for r in recs if r.get("stop_reason") == "sql_close") / n, 4
        )
        if n
        else None,
        "truncation_rate": round(
            sum(1 for r in recs if r.get("stop_reason") == "max_new_tokens") / n, 4
        )
        if n
        else None,
        "multiple_sql_block_rate": round(
            sum(1 for r in recs if r["raw_completion"].lower().count("<sql>") > 1) / n, 4
        )
        if n
        else None,
        "mean_completion_tokens": round(
            sum(r.get("completion_token_count", 0) for r in recs) / n, 2
        )
        if n
        else None,
        "mode_counts": {
            mode: sum(1 for r in recs if r.get("mode") == mode)
            for mode in sorted({r.get("mode") for r in recs})
        },
        "per_mode": {},
    }
    for mode in metrics["mode_counts"]:
        sub = [r for r in recs if r.get("mode") == mode]
        m = len(sub)
        metrics["per_mode"][mode] = {
            "records": m,
            "format_valid_rate": round(sum(1 for r in sub if r.get("format_valid")) / m, 4)
            if m
            else None,
            "parse_valid_rate": round(sum(1 for r in sub if r.get("sql_parse_valid")) / m, 4)
            if m
            else None,
            "safe_rate": round(sum(1 for r in sub if r.get("safe")) / m, 4) if m else None,
            "execution_success_rate": round(
                sum(1 for r in sub if r.get("execution_success")) / m, 4
            )
            if m
            else None,
            "strict_execution_accuracy": round(
                sum(1 for r in sub if r.get("strict_equivalent")) / m, 4
            )
            if m
            else None,
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = compute(args.input)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
