#!/usr/bin/env python3
"""Validation diagnostic on validation_tuning (32 examples, no Mini-Dev).

Saves raw completions for deterministic (do_sample=False) and sampled modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import StoppingCriteriaList

from querydistill.data.schema import load_examples
from querydistill.evaluation.modelspec import ModelSpec, load_model
from querydistill.generation.stopping import StopAfterSqlClose
from querydistill.outputs.parser import parse_model_output
from querydistill.outputs.prompting import build_prompt
from querydistill.sql.environment import SQLExecutionEnvironment
from querydistill.sql.safety import validate_sql
from querydistill.sql.verifier import ResultEquivalenceVerifier
from querydistill.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--quantized-model", default=None)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    args = parser.parse_args()

    if args.quantized_model:
        spec = ModelSpec(
            stage="gptq",
            base_model_path=args.base_model,
            quantized_model_path=args.quantized_model,
        )
    else:
        spec = ModelSpec(
            stage="adapter" if args.adapter else "base",
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
    import torch

    examples = load_examples(args.examples)
    ids = load_json(args.ids)
    id_set = {e["example_id"] for e in ids["examples"]}
    selected = [e for e in examples if e.example_id in id_set]
    environment = SQLExecutionEnvironment.from_registry(args.registry)
    verifier = ResultEquivalenceVerifier()

    records = []
    for example in selected:
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
        for mode, do_sample, temperature, top_p in [
            ("deterministic", False, None, None),
            ("sampled", True, 1.0, 0.95),
        ]:
            torch.manual_seed(args.seed + len(records))
            stop_criteria = StopAfterSqlClose(
                tokenizer, prompt_length=inputs["input_ids"].shape[-1]
            )
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
            )
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            outputs = model.generate(**inputs, **gen_kwargs)
            generated_token_ids = outputs[0][inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            stop_reason = (
                "sql_close"
                if len(generated_token_ids) >= len(stop_criteria.stop_ids)
                and generated_token_ids[-len(stop_criteria.stop_ids) :].tolist()
                == stop_criteria.stop_ids
                else "max_new_tokens"
            )
            parsed = parse_model_output(raw, require_plan=False)
            decision = validate_sql(parsed.sql) if parsed.sql else None
            safe = bool(decision and decision.safe)
            execution_success = False
            strict_equivalent = False
            error_type = "none"
            if safe and parsed.sql:
                candidate = environment.execute(example.db_id, parsed.sql)
                gold = environment.execute(example.db_id, example.gold_sql)
                execution_success = candidate.success
                verification = verifier.verify(
                    candidate=candidate,
                    gold=gold,
                    candidate_sql=parsed.sql,
                    gold_sql=example.gold_sql,
                    schema_tables=set(environment.table_names(example.db_id)),
                )
                strict_equivalent = verification.strict_equivalent
                error_type = verification.kind if not strict_equivalent else "none"
            records.append(
                {
                    "example_id": example.example_id,
                    "db_id": example.db_id,
                    "mode": mode,
                    "raw_completion": raw,
                    "completion_token_count": len(generated_token_ids),
                    "finish_reason": "generated",
                    "stop_reason": stop_reason,
                    "format_valid": bool(parsed.sql),
                    "parsed_sql": parsed.sql,
                    "sql_parse_valid": bool(parsed.sql) and not parsed.parse_error,
                    "safe": safe,
                    "execution_success": execution_success,
                    "strict_equivalent": strict_equivalent,
                    "error_type": error_type,
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
