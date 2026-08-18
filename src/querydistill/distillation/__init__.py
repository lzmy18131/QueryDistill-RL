"""Teacher backends for offline distillation candidate generation."""

from .backends import MockTeacherBackend, TeacherBackend, TransformersTeacherBackend
from .pipeline import DistillationConfig, DistillationPipeline

__all__ = [
    "MockTeacherBackend",
    "TeacherBackend",
    "TransformersTeacherBackend",
    "DistillationConfig",
    "DistillationPipeline",
]
