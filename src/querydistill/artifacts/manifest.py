"""Stage artifact manifests for the experiment chain.

Every training/quantization stage writes ``artifact_manifest.json`` so the
chain Base -> SFT -> GRPO -> merge -> GPTQ -> vLLM is auditable:

* which input artifact was consumed (with sha256)
* which output artifact was produced
* base model / adapter identity
* config hash and creation time
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..utils import atomic_write_json, sha256_file, utc_now_iso


@dataclass
class ArtifactManifest:
    stage: str
    input_artifact: str | None = None
    input_sha256: str | None = None
    output_artifact: str | None = None
    output_sha256: str | None = None
    base_model: str | None = None
    adapter: str | None = None
    config_hash: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def write(self, directory: str | Path, filename: str = "artifact_manifest.json") -> Path:
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self.as_dict())
        return path


def config_hash(payload: dict | object) -> str:
    if hasattr(payload, "__dict__"):
        payload = payload.__dict__
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_or_none(path: str | Path | None) -> str | None:
    return str(path) if path else None


def hash_or_none(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    return sha256_file(path) if path.exists() else None
