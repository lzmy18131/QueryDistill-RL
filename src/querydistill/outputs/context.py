"""Fail-closed model context shared by real teacher / evaluation backends.

Real model backends may only ever receive example identity + split metadata.
Gold SQL, gold results, and expected answers are forbidden by construction
here and are available only to test-only oracle doubles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RealModelContext:
    example_id: str
    db_id: str
    split: str

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def safe_keys(cls) -> frozenset[str]:
        return frozenset({"example_id", "db_id", "split"})
