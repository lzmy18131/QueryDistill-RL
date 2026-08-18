#!/usr/bin/env python3
"""Generate BIRD_DATA_REPORT.md from the prepared BIRD pilot subset.

Usage:
  python tools/generate_bird_data_report.py --examples data/bird/examples --registry data/bird/db_registry.json --tokenizer models/qwen3-0.6b-base --out docs/BIRD_DATA_REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from querydistill.data.schema import load_examples
from querydistill.outputs.prompting import apply_student_chat_template, build_prompt
from querydistill.sql.environment import SQLExecutionEnvironment


def _len_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default="data/bird/examples")
    parser.add_argument("--registry", default="data/bird/db_registry.json")
    parser.add_argument("--tokenizer", default="models/qwen3-0.6b-base")
    parser.add_argument("--out", default="docs/BIRD_DATA_REPORT.md")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    env = SQLExecutionEnvironment.from_registry(args.registry)

    train = load_examples(Path(args.examples) / "train.jsonl")
    dev = load_examples(Path(args.examples) / "dev.jsonl")
    all_examples = train + dev

    db_ids = sorted({e.db_id for e in all_examples})
    schema_tokens = []
    evidence_tokens = []
    prompt_tokens = []
    chat_prompt_tokens = []
    truncated = {512: 0, 768: 0, 1024: 0, 1536: 0}

    for example in all_examples:
        schema_tokens.append(_len_tokens(tokenizer, example.schema_text))
        evidence_tokens.append(_len_tokens(tokenizer, example.evidence) if example.evidence else 0)
        raw_prompt = build_prompt(
            example.question,
            example.schema_text,
            example.db_id,
            include_plan=False,
            evidence=example.evidence,
        )
        prompt_tokens.append(_len_tokens(tokenizer, raw_prompt))
        chat_prompt = apply_student_chat_template(
            tokenizer,
            example.question,
            example.schema_text,
            example.db_id,
            include_plan=False,
            evidence=example.evidence,
            tokenize=False,
        )
        chat_len = _len_tokens(tokenizer, chat_prompt)
        chat_prompt_tokens.append(chat_len)
        for threshold in truncated:
            if chat_len > threshold:
                truncated[threshold] += 1

    gold_success = 0
    gold_total = 0
    gold_failures = []
    for example in all_examples:
        result = env.execute(example.db_id, example.gold_sql)
        gold_total += 1
        if result.success:
            gold_success += 1
        else:
            gold_failures.append({"example_id": example.example_id, "error": result.error_type})

    def pct(arr, p):
        if not arr:
            return 0
        ordered = sorted(arr)
        idx = max(0, min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1)))))
        return ordered[idx]

    lines = [
        "# BIRD Data Report (Pilot Subset)",
        "",
        f"- Generated: {json.dumps({'source': 'birdsql/bird23-train-filtered + bird_mini_dev'})}",
        f"- Train count: {len(train)}",
        f"- Dev count: {len(dev)}",
        "- Test count: 0",
        f"- Database count: {len(db_ids)}",
        f"- Databases: {', '.join(db_ids)}",
        f"- Gold execution success: {gold_success}/{gold_total}",
        "",
        "## Token statistics",
        "",
        "| Metric | P50 | P90 | P95 | P99 | Max |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Schema tokens | {pct(schema_tokens, 50)} | {pct(schema_tokens, 90)} | {pct(schema_tokens, 95)} | {pct(schema_tokens, 99)} | {max(schema_tokens)} |",
        f"| Evidence tokens | {pct(evidence_tokens, 50)} | {pct(evidence_tokens, 90)} | {pct(evidence_tokens, 95)} | {pct(evidence_tokens, 99)} | {max(evidence_tokens)} |",
        f"| Raw prompt tokens | {pct(prompt_tokens, 50)} | {pct(prompt_tokens, 90)} | {pct(prompt_tokens, 95)} | {pct(prompt_tokens, 99)} | {max(prompt_tokens)} |",
        f"| Chat-serialized prompt tokens | {pct(chat_prompt_tokens, 50)} | {pct(chat_prompt_tokens, 90)} | {pct(chat_prompt_tokens, 95)} | {pct(chat_prompt_tokens, 99)} | {max(chat_prompt_tokens)} |",
        "",
        "## Truncation rate by max_prompt_length",
        "",
        "| max_prompt_length | examples over budget | rate |",
        "| --- | --- | --- |",
    ]
    for threshold in (512, 768, 1024, 1536):
        lines.append(
            f"| {threshold} | {truncated[threshold]} | {truncated[threshold] / len(all_examples):.2%} |"
        )
    if gold_failures:
        lines.append("", "## Gold failures", "")
        for fail in gold_failures:
            lines.append(f"- {fail['example_id']}: {fail['error']}")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append(
        "This report is generated from the real BIRD pilot subset; it is not a benchmark claim."
    )
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
