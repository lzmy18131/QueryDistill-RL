"""Data-layer primitives: schema, fixtures, leakage guard, audit, adapters."""

from .leakage import LeakageGuard, LeakageReport
from .schema import DistillationRecord, Example, load_distillation_records, load_examples
from .splitter import create_train_validation_split

__all__ = [
    "LeakageGuard",
    "LeakageReport",
    "DistillationRecord",
    "Example",
    "load_distillation_records",
    "load_examples",
    "create_train_validation_split",
]
