#!/usr/bin/env python3
"""Generation-only diagnostic for Distilled-SFT before GRPO.

Runs 24 stratified validation_tuning prompts x 4 completions with the future
GRPO sampling configuration. No backward, no optimizer, no policy update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import StoppingCriteriaList

from querydistill.data.schema import load_examples
from querydistill.evaluation.modelspec import ModelSpec, load_model
from querydistill.generation.stopping import StopAfterSqlClose
from querydistill.outputs.fallback import extract_fallback_sql
from querydistill.outputs.parser import parse_model_output
from querydistill.outputs.prompting import build_prompt
from querydistill.rewards.composite import CompositeReward, GoldResultCache
from querydistill.sql.environment import SQLExecutionEnvironment
from querydistill.sql.safety import validate_sql
from querydistill.sql.verifier import ResultEquivalenceVerifier
from querydistill.utils import atomic_write_json, load_json, sha256_file, utc_now_iso


def _reward_histogram(records: list[dict], bucket_size: float = 0.1) -> dict[str, int]:
    """Return a small reward histogram keyed by bucket lower bound."""
    histogram: dict[str, int] = {}
    for record in records:
        reward = record.get("total_reward")
        if reward is None:
            bucket = "None"
        else:
            bucket = f"{round(float(reward) // bucket_size * bucket_size, 2):.2f}"
        histogram[bucket] = histogram.get(bucket, 0) + 1
    return dict(sorted(histogram.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    spec = ModelSpec(
        stage="adapter",
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    load_kwargs = {}
    if args.device:
        load_kwargs["device_map"] = args.device
    if args.dtype:
        import torch

        load_kwargs["torch_dtype"] = getattr(torch, args.dtype)
    model, tokenizer = load_model(spec, **load_kwargs)
    model.eval()
    # Qwen chat templates use <|im_end|>; the base config may still point at
    # <|endoftext|> so make generation stop at the chat end token.
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    examples = load_examples(args.examples)
    ids = load_json(args.ids)
    id_set = {e["example_id"] for e in ids["examples"]}
    selected = [e for e in examples if e.example_id in id_set]
    environment = SQLExecutionEnvironment.from_registry(args.registry)
    dataset_hash = sha256_file(Path(args.examples))
    db_fingerprints = {db: sha256_file(path) for db, path in environment.db_paths.items()}
    cache = GoldResultCache(
        cache_dir=Path(args.output).parent / "gold_cache",
        dataset_hash=dataset_hash,
        database_fingerprints=db_fingerprints,
    )
    reward = CompositeReward(environment=environment, gold_cache=cache)
    verifier = ResultEquivalenceVerifier()

    import torch

    records = []
    for group_id, example in enumerate(selected, start=1):
        prompt = build_prompt(
            question=example.question,
            schema_text=example.schema_text,
            db_id=example.db_id,
            include_plan=False,
            evidence=example.evidence,
        )
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        torch.manual_seed(args.seed + group_id)
        stop_criteria = StopAfterSqlClose(tokenizer, prompt_length=inputs["input_ids"].shape[-1])
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_completions,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
        )
        for idx in range(args.num_completions):
            generated_token_ids = outputs.sequences[idx][inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            stop_reason = (
                "sql_close"
                if len(generated_token_ids) >= len(stop_criteria.stop_ids)
                and generated_token_ids[-len(stop_criteria.stop_ids) :].tolist()
                == stop_criteria.stop_ids
                else "max_new_tokens"
            )
            parsed = parse_model_output(raw, require_plan=False)
            strict_format_valid = bool(parsed.sql)
            fallback_sql = extract_fallback_sql(raw)
            fallback_parsed = (
                parse_model_output(f"<sql>\n{fallback_sql}\n</sql>", require_plan=False)
                if fallback_sql
                else None
            )
            sql_to_score = parsed.sql or (fallback_parsed.sql if fallback_parsed else None)
            decision = validate_sql(sql_to_score) if sql_to_score else None
            safe = bool(decision and decision.safe)
            execution_success = False
            strict_equivalent = False
            if safe and sql_to_score:
                candidate = environment.execute(example.db_id, sql_to_score)
                gold = environment.execute(example.db_id, example.gold_sql)
                execution_success = candidate.success
                verification = verifier.verify(
                    candidate=candidate,
                    gold=gold,
                    candidate_sql=sql_to_score,
                    gold_sql=example.gold_sql,
                    schema_tables=set(environment.table_names(example.db_id)),
                )
                strict_equivalent = verification.strict_equivalent
            trace = reward.score_once(example, raw) if sql_to_score else None
            record = {
                "group_id": f"group-{group_id}",
                "example_id": example.example_id,
                "db_id": example.db_id,
                "completion_index": idx,
                "raw_completion": raw,
                "normalized_completion_hash": hashlib.sha256(raw.encode()).hexdigest(),
                "strict_format_valid": strict_format_valid,
                "strict_parsed_sql": parsed.sql,
                "diagnostic_fallback_sql": fallback_sql,
                "fallback_parse_valid": bool(fallback_parsed and fallback_parsed.sql),
                "safe": safe,
                "execution_success": execution_success,
                "strict_equivalent": strict_equivalent,
                "reward_components": trace.breakdown.as_dict() if trace else None,
                "total_reward": trace.breakdown.total if trace else None,
                "completion_tokens": len(generated_token_ids),
                "finish_reason": "generated",
                "stop_reason": stop_reason,
            }
            records.append(record)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    groups = {}
    for r in records:
        groups.setdefault(r["group_id"], []).append(r)
    group_stds = []
    for _gid, items in groups.items():
        rewards = [r["total_reward"] or 0.0 for r in items]
        mean = sum(rewards) / len(rewards)
        std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
        group_stds.append(std)

    metrics = {
        "groups": len(groups),
        "completions": len(records),
        "raw_completion_unique_ratio": round(
            len({r["normalized_completion_hash"] for r in records}) / len(records), 4
        ),
        "normalized_sql_unique_ratio": round(
            len({r["strict_parsed_sql"] or r["diagnostic_fallback_sql"] for r in records})
            / len(records),
            4,
        ),
        "format_valid_rate": round(
            sum(1 for r in records if r["strict_format_valid"]) / len(records), 4
        ),
        "parse_valid_rate": round(
            sum(1 for r in records if r["strict_parsed_sql"]) / len(records), 4
        ),
        "safe_rate": round(sum(1 for r in records if r["safe"]) / len(records), 4),
        "execution_success_rate": round(
            sum(1 for r in records if r["execution_success"]) / len(records), 4
        ),
        "strict_correct_rate": round(
            sum(1 for r in records if r["strict_equivalent"]) / len(records), 4
        ),
        "nonzero_reward_std_group_count": sum(1 for s in group_stds if s > 1e-9),
        "nonzero_reward_std_group_fraction": round(
            sum(1 for s in group_stds if s > 1e-9) / len(group_stds), 4
        ),
        "group_reward_stds": [round(s, 6) for s in group_stds],
        "mean_completion_tokens": round(
            sum(r["completion_tokens"] for r in records) / len(records), 2
        ),
        "stop_at_sql_close_rate": round(
            sum(1 for r in records if r.get("stop_reason") == "sql_close") / len(records), 4
        ),
        "truncation_rate": round(
            sum(1 for r in records if r.get("stop_reason") == "max_new_tokens") / len(records), 4
        ),
        "multiple_sql_block_rate": round(
            sum(1 for r in records if r["raw_completion"].lower().count("<sql>") > 1)
            / len(records),
            4,
        ),
        "format_invalid_but_sql_extractable_rate": round(
            sum(
                1
                for r in records
                if not r["strict_format_valid"]
                and r.get("diagnostic_fallback_sql")
                and r.get("fallback_parse_valid")
            )
            / sum(1 for r in records if not r["strict_format_valid"])
            if any(not r["strict_format_valid"] for r in records)
            else 0.0,
            4,
        ),
        "reward_histogram": _reward_histogram(records),
        "created_at": utc_now_iso(),
    }
    atomic_write_json(args.metrics_output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
