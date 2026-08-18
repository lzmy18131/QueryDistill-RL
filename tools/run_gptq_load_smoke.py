#!/usr/bin/env python3
"""GPTQ load smoke: load existing INT4 checkpoint and generate 1 validation sample.

Only load/validation; never re-quantizes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from querydistill.data.schema import load_examples
from querydistill.generation.stopping import StopAfterSqlClose
from querydistill.outputs.parser import parse_model_output
from querydistill.outputs.prompting import build_prompt
from querydistill.utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized-model", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    quantized = Path(args.quantized_model)
    load_start = time.time()
    try:
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            str(quantized),
            device_map=args.device,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(quantized))
        load_seconds = time.time() - load_start
    except Exception as exc:  # noqa: BLE001
        atomic_write_json(
            args.output,
            {
                "load_ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
                "load_seconds": time.time() - load_start,
            },
        )
        print(f"LOAD_FAIL {type(exc).__name__}: {exc}")
        return

    examples = load_examples(args.examples)
    example = next(e for e in examples if e.example_id == args.example_id)
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
    stop_criteria = StopAfterSqlClose(tokenizer, prompt_length=inputs["input_ids"].shape[-1])
    gen_start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )
    gen_seconds = time.time() - gen_start
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = parse_model_output(raw, require_plan=False)
    result = {
        "load_ok": True,
        "model_path": str(quantized),
        "model_size_bytes": sum(p.stat().st_size for p in quantized.rglob("*") if p.is_file()),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(gen_seconds, 3),
        "completion_tokens": len(generated),
        "stop_reason": (
            "sql_close"
            if len(generated) >= len(stop_criteria.stop_ids)
            and generated[-len(stop_criteria.stop_ids) :].tolist() == stop_criteria.stop_ids
            else "max_new_tokens"
        ),
        "raw_completion": raw,
        "format_valid": bool(parsed.sql),
        "parsed_sql": parsed.sql,
        "example_id": example.example_id,
        "db_id": example.db_id,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
