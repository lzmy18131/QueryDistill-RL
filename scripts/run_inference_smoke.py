#!/usr/bin/env python3
"""Student base inference smoke (GPU).

Loads the student locally, runs one tiny-fixture prompt, parses the output
with the real protocol parser and scores it through the real SQLite reward
environment. PASS means the inference path executed end-to-end; it makes no
accuracy claim for the untrained base model.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from querydistill.config import Settings
from querydistill.data.schema import load_examples
from querydistill.outputs.prompting import build_prompt
from querydistill.rewards.composite import CompositeReward
from querydistill.sql.environment import SQLExecutionEnvironment
from querydistill.utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--examples", default="data/tiny_sql/examples.jsonl")
    parser.add_argument("--registry", default="tests/fixtures/tiny_sql/db_registry.json")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output-dir", default="artifacts/smoke/inference")
    args = parser.parse_args()

    settings = Settings.load()
    project = settings.project_root
    model_path = (project / args.model_path).resolve()
    examples_path = (project / args.examples).resolve()
    registry_path = (project / args.registry).resolve()
    output_dir = (project / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status = {"status": "BLOCKED", "model_path": str(model_path)}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            status["reason"] = "CUDA not available to torch"
            atomic_write_json(output_dir / "status.json", status)
            print(json.dumps(status, indent=2))
            return

        from querydistill.evaluation.harness import EvaluationHarness, TransformersModelBackend

        examples = load_examples(examples_path)
        eval_split = "test"
        example = next(e for e in examples if e.split == eval_split)
        environment = SQLExecutionEnvironment.from_registry(registry_path)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        backend = TransformersModelBackend(str(model_path), max_new_tokens=args.max_new_tokens)
        # Reuse the already-loaded model/tokenizer inside the backend to avoid a
        # second allocation.
        backend._model = model
        backend._tokenizer = tokenizer

        record = None
        composite = CompositeReward(environment, require_plan=True)

        # Generate exactly once for this smoke. The same completion is then used
        # by both the evaluation harness and the reward path; no second model
        # generation is performed.
        prompt = build_prompt(
            example.question, example.schema_text, example.db_id, include_plan=True
        )
        generation_started = time.monotonic()
        output = backend.generate(
            prompt,
            context={
                "example_id": example.example_id,
                "db_id": example.db_id,
                "split": example.split,
            },
        )
        generation_latency_ms = (time.monotonic() - generation_started) * 1000.0

        class _SingleGenerationBackend:
            name = backend.name

            def generate(self, current_prompt, context=None):
                if current_prompt != prompt:
                    raise AssertionError("inference smoke must generate exactly once")
                return output

        harness = EvaluationHarness(environment, _SingleGenerationBackend(), require_plan=True)
        metrics = harness.run([example], split=eval_split)
        record = metrics.records[0]
        breakdown = composite.evaluate(example, output)

        status = {
            "status": "PASS",
            "example_id": example.example_id,
            "execution_equivalent": record.execution_equivalent,
            "error_bucket": record.error_bucket,
            "reward_total": breakdown.total,
            "generation_latency_ms": round(generation_latency_ms, 2),
            "cached_eval_latency_ms": record.latency_ms,
            "note": "PASS = inference pipeline executed; base-model accuracy is not claimed",
        }
        atomic_write_json(
            output_dir / "environment.json",
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "vram_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            },
        )
        atomic_write_json(output_dir / "status.json", status)
        atomic_write_text_md(output_dir, status)
        print(json.dumps(status, indent=2))
    except Exception as exc:  # noqa: BLE001 - recorded, not faked
        status["reason"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output_dir / "status.json", status)
        atomic_write_text_md(output_dir, status)
        print(json.dumps(status, indent=2))
    finally:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def atomic_write_text_md(output_dir: Path, payload: dict) -> None:
    lines = ["# Student base inference smoke", "", f"STATUS: {payload.get('status')}", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
