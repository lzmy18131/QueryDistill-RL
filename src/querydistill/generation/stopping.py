"""Stopping criteria that stop generation exactly after the first ``</sql>``.

The criteria intentionally checks only the generated completion portion, not
the prompt.  A stateful per-sequence ``_finished`` mask also prevents a
sequence from generating a second SQL block after it has already emitted a
closing tag.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import StoppingCriteria

SQL_CLOSE_TAG = "</sql>"


class StopAfterSqlClose(StoppingCriteria):
    """Stop each active sequence after it emits the ``</sql>`` terminator.

    Parameters
    ----------
    tokenizer:
        A HF tokenizer (or tokenizer-like object) used to pre-tokenize the
        closing marker.  The object must support
        ``tokenizer("</sql>", add_special_tokens=False).input_ids``.
    prompt_length:
        Number of prompt tokens in each sequence.  May be an ``int`` when all
        sequences share the same prompt length, or a list/tensor with one
        value per sequence for padded batches with different prompt lengths.
    """

    def __init__(
        self, tokenizer, prompt_length: int | Sequence[int] | torch.Tensor | None = None
    ) -> None:
        self.stop_ids = list(tokenizer(SQL_CLOSE_TAG, add_special_tokens=False).input_ids)
        if not self.stop_ids:
            raise ValueError("tokenizer produced empty </sql> token ids")
        if prompt_length is None:
            self.prompt_length: torch.Tensor | int | None = None
        elif isinstance(prompt_length, torch.Tensor):
            self.prompt_length = prompt_length.to(dtype=torch.long)
        elif isinstance(prompt_length, int):
            self.prompt_length = prompt_length
        else:
            self.prompt_length = torch.as_tensor(list(prompt_length), dtype=torch.long)
        self._finished: torch.Tensor | None = None

    def _start(self, batch_size: int, device: torch.device) -> None:
        if self._finished is None or self._finished.shape[0] != batch_size:
            self._finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _row_generated(self, row: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Return the generated portion for one row, ignoring the prompt."""
        if prompt_len <= 0:
            return row
        if row.shape[-1] <= prompt_len:
            return row.new_empty(0)
        return row[prompt_len:]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        self._start(batch_size, device)
        if self._finished is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)

        stop_tensor = torch.tensor(self.stop_ids, dtype=input_ids.dtype, device=device)
        n = len(self.stop_ids)
        for i in range(batch_size):
            if bool(self._finished[i].item()):
                continue
            if isinstance(self.prompt_length, torch.Tensor):
                if self.prompt_length.numel() == 1:
                    plen = int(self.prompt_length.item())
                else:
                    plen = int(self.prompt_length[i].item())
            elif isinstance(self.prompt_length, int):
                plen = self.prompt_length
            else:
                plen = 0  # Legacy mode: check the full sequence.
            gen = self._row_generated(input_ids[i], plen)
            if gen.shape[-1] < n:
                continue
            if bool((gen[-n:] == stop_tensor).all().item()):
                self._finished[i] = True
        return self._finished.clone()


__all__ = ["SQL_CLOSE_TAG", "StopAfterSqlClose"]
