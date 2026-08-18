#!/usr/bin/env python3
"""OpenAI-compatible vLLM benchmark client (TTFT / latency / tokens / VRAM).

Usage:
  python scripts/benchmark_vllm.py --prompts-file prompts.jsonl --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from querydistill.serving.vllm import benchmark, save_benchmark
from querydistill.utils import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="querydistill")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="artifacts/benchmark/vllm_report.json")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat prompt list N times")
    parser.add_argument("--api-key", default=None, help="vLLM API key (or VLLM_API_KEY env)")
    args = parser.parse_args()

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        for record in load_jsonl(Path(args.prompts_file)):
            prompts.append(str(record["prompt"]))
    if not prompts:
        raise SystemExit("provide --prompt or --prompts-file")
    prompts = prompts * args.repeat

    report = benchmark(
        prompts=prompts,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        api_key=args.api_key or os.environ.get("VLLM_API_KEY"),
    )
    save_benchmark(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
