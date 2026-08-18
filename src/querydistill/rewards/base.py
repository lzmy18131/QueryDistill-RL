"""Shared reward types.

Component ranges are deliberately small so that the correctness component
dominates every other signal:

* format / parse / safety / execution positives: +0.05 each (max +0.20)
* execution equivalence: +1.00
* unsafe SQL: hard -1.00 (total is clamped at -1.0)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RewardBreakdown:
    format: float = 0.0
    parse: float = 0.0
    safety: float = 0.0
    execution: float = 0.0
    correctness: float = 0.0
    total: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> RewardBreakdown:
        return cls(
            format=float(payload.get("format", 0.0)),
            parse=float(payload.get("parse", 0.0)),
            safety=float(payload.get("safety", 0.0)),
            execution=float(payload.get("execution", 0.0)),
            correctness=float(payload.get("correctness", 0.0)),
            total=float(payload.get("total", 0.0)),
            notes=dict(payload.get("notes", {})),
        )
