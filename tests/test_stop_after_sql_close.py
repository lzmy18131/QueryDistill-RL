"""Tests for StopAfterSqlClose generation stopping criteria.

The criteria must:
- inspect only the generated completion portion, never the prompt;
- stop exactly after the first ``</sql>`` terminator;
- work per-sequence in a batch;
- remain non-stopping when no closing tag has been generated.
"""

from __future__ import annotations

import torch

from querydistill.generation.stopping import StopAfterSqlClose

# A deliberately unusual tokenization of "</sql>" for deterministic tests.
CLOSE_IDS = [42, 43]


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        return _FakeEncoding([42, 43] if "</sql>" in text else [])


class _FakeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


def make_input(prompt_ids, generated_ids):
    """Build a padded batch tensor from per-row (prompt, generated) token lists."""
    rows = [list(prompt) + list(gen) for prompt, gen in zip(prompt_ids, generated_ids, strict=True)]
    max_len = max(len(r) for r in rows)
    padded = [r + [0] * (max_len - len(r)) for r in rows]
    return torch.tensor(padded, dtype=torch.long)


def test_prompt_contains_close_does_not_stop_before_generation():
    # Prompt itself ends with </sql>; no generated tokens yet.
    tok = FakeTokenizer()
    crit = StopAfterSqlClose(tok, prompt_length=6)
    input_ids = make_input([[1, 2, 42, 43, 3, 4]], [[]])
    result = crit(input_ids, None)
    assert result.tolist() == [False]


def test_stop_after_first_close_in_generated_region():
    tok = FakeTokenizer()
    crit = StopAfterSqlClose(tok, prompt_length=4)
    input_ids = make_input([[1, 2, 3, 4]], [[10, 11, 42, 43]])
    result = crit(input_ids, None)
    assert result.tolist() == [True]


def test_no_close_does_not_stop():
    tok = FakeTokenizer()
    crit = StopAfterSqlClose(tok, prompt_length=4)
    input_ids = make_input([[1, 2, 3, 4]], [[10, 11, 12]])
    result = crit(input_ids, None)
    assert result.tolist() == [False]


def test_batch_different_finish_times():
    tok = FakeTokenizer()
    crit = StopAfterSqlClose(tok, prompt_length=4)
    input_ids = make_input(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[10, 11, 42, 43], [20, 21, 22, 23]],
    )
    result = crit(input_ids, None)
    assert result.tolist() == [True, False]


def test_prompt_contains_close_but_generated_not_close_still_running():
    tok = FakeTokenizer()
    crit = StopAfterSqlClose(tok, prompt_length=6)
    # Prompt contains the closing token ids at positions 2..3, but the tail of
    # the generated region does not.
    input_ids = make_input([[1, 2, 42, 43, 3, 4]], [[5, 6, 7]])
    result = crit(input_ids, None)
    assert result.tolist() == [False]


def test_stateful_once_finished_stays_finished():
    tok = FakeTokenizer()
    crit = StopAfterSqlClose(tok, prompt_length=4)
    # First call: close is at the tail.
    finished_input = make_input([[1, 2, 3, 4]], [[10, 42, 43]])
    assert crit(finished_input, None).tolist() == [True]
    # A later call should keep the row finished even if more tokens were
    # appended (defensive against non-streaming call patterns).
    later_input = make_input([[1, 2, 3, 4]], [[10, 42, 43, 99, 100]])
    assert crit(later_input, None).tolist() == [True]


def test_prompt_length_tensor_per_row():
    tok = FakeTokenizer()
    prompt_lengths = torch.tensor([4, 5])
    crit = StopAfterSqlClose(tok, prompt_length=prompt_lengths)
    # Row 0 has a padded prompt (actual length 4, padded to 5); both rows have
    # the same total length so there is no padding after the generated region.
    input_ids = make_input(
        [[1, 2, 3, 4, 0], [5, 6, 7, 8, 9]],
        [[10, 42, 43], [11, 42, 43]],
    )
    result = crit(input_ids, None)
    assert result.tolist() == [True, True]
