"""Unified evaluation harness for Base / Gold-SFT / Distilled-SFT / GRPO / GPTQ."""

from .errors import ErrorBucket, classify_error
from .harness import EvaluationHarness, MockModelBackend, ModelBackend
from .metrics import EvaluationMetrics, EvaluationRecord
from .statistics import mcnemar_exact, paired_bootstrap_ci

__all__ = [
    "ErrorBucket",
    "classify_error",
    "EvaluationHarness",
    "MockModelBackend",
    "ModelBackend",
    "EvaluationMetrics",
    "EvaluationRecord",
    "mcnemar_exact",
    "paired_bootstrap_ci",
]
