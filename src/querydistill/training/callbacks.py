"""Reward logging for GRPO and tiny training status helpers.

The GRPO optimizer itself comes from TRL. What this project owns is the
**real** SQLite reward environment and the logging that makes every reward
component auditable (``reward_samples.jsonl``).
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..utils import append_jsonl, utc_now_iso


class RewardSampleLogger:
    """Wraps a TRL reward function and persists every scored rollout.

    By default records are compact (no full prompt/completion/result rows).
    Set ``debug_full_trace=True`` to include the full trace dict.
    """

    def __init__(
        self,
        reward_fn,
        path: str | Path,
        registry,
        debug_full_trace: bool = False,
        run_id: str | None = None,
    ):
        self._reward_fn = reward_fn
        self.path = Path(path)
        self.registry = registry
        self.debug_full_trace = debug_full_trace
        self.run_id = run_id
        self._lock = threading.Lock()
        self._count = 0
        self._generation_batch_counter = 0

    def __call__(self, completions, prompts=None, **kwargs):
        with self._lock:
            self._generation_batch_counter += 1
            generation_group_id = f"gen-{self._generation_batch_counter}"
        rewards = self._reward_fn(completions=completions, prompts=prompts, **kwargs)
        prompts = prompts or []
        # score_once traces are captured by SQLRewardFunction during scoring;
        # re-executing SQL here is forbidden.
        traces = getattr(self._reward_fn, "traces", [])
        records = []
        for index, completion in enumerate(completions):
            prompt = prompts[index] if index < len(prompts) else ""
            example_ids = kwargs.get("example_id")
            example_id = (
                str(example_ids[index])
                if example_ids is not None and index < len(example_ids)
                else None
            )
            example = self.registry.get(example_id) if example_id else None
            trace_obj = (
                traces[-len(completions) + index] if len(traces) >= len(completions) else None
            )
            trace_dict = trace_obj.as_dict() if trace_obj else None
            breakdown = trace_obj.breakdown if trace_obj else None
            verification = trace_obj.verification if trace_obj else None
            candidate = trace_obj.candidate_execution if trace_obj else None
            gold = trace_obj.gold_execution if trace_obj else None
            record = {
                "timestamp": utc_now_iso(),
                "run_id": self.run_id,
                "generation_group_id": generation_group_id,
                "example_id": example_id or getattr(example, "example_id", None),
                "db_id": getattr(example, "db_id", None),
                "reward": float(rewards[index]) if index < len(rewards) else None,
                "format_reward": breakdown.format if breakdown else None,
                "parse_reward": breakdown.parse if breakdown else None,
                "safety_reward": breakdown.safety if breakdown else None,
                "execution_reward": breakdown.execution if breakdown else None,
                "correctness_reward": breakdown.correctness if breakdown else None,
                "parse_ok": (
                    bool(trace_obj.parse.sql is not None and not trace_obj.parse.parse_error)
                    if trace_obj
                    else None
                ),
                "safe": bool(trace_obj.safety.safe) if trace_obj and trace_obj.safety else None,
                "execution_success": bool(candidate.success) if candidate else None,
                "strict_equivalent": bool(verification.strict_equivalent) if verification else None,
                "partial_credit": bool(verification.partial_credit) if verification else None,
                "error_type": (
                    verification.kind
                    if verification
                    else (candidate.error_type if candidate else None)
                ),
                "candidate_row_count": candidate.row_count if candidate else None,
                "gold_row_count": gold.row_count if gold else None,
            }
            if self.debug_full_trace:
                record["prompt"] = prompt
                record["completion"] = (
                    completion if isinstance(completion, str) else repr(completion)
                )
                record["trace"] = trace_dict
            records.append(record)
        with self._lock:
            append_jsonl(self.path, records)
            self._count += len(records)
        # The trace buffer is only needed for the current batch; clear it so a
        # long GRPO run does not retain every RewardTrace in memory.
        clear = getattr(self._reward_fn, "clear_traces", None)
        if callable(clear):
            clear()
        return rewards

    @property
    def count(self) -> int:
        return self._count


def write_smoke_readme(path: Path, name: str, status: str, details: dict) -> None:
    from ..utils import atomic_write_text

    lines = [f"# {name}", "", f"STATUS: {status}", ""]
    for key, value in details.items():
        lines.append(f"- {key}: {value}")
    atomic_write_text(path, "\n".join(lines) + "\n")
