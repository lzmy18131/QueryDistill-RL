"""Configuration system.

Design goals (first implementation round):

* Every path used by training/downloading/quantization flows through this module.
* No hard-coded user names. No default C-drive caches.
* The application never mutates the user's system environment; it only builds
  environment dictionaries for its own subprocesses.
* WSL-style ``/mnt/d/...`` paths written in ``.env`` are translated to
  ``D:\\...`` when the code itself runs on native Windows, so one ``.env``
  example works for both shells.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_REPO_MARKER = "querydistill-rl"

# (env var name, cache key attribute, fallback subdir under the cache root)
CACHE_ENV_VARS = {
    "HF_HOME": "hf_home",
    "HF_DATASETS_CACHE": "hf_datasets_cache",
    "TORCH_HOME": "torch_home",
    "XDG_CACHE_HOME": "xdg_cache_home",
    "UV_CACHE_DIR": "uv_cache_dir",
    "PIP_CACHE_DIR": "pip_cache_dir",
    "TRITON_CACHE_DIR": "triton_cache_dir",
    "TORCH_EXTENSIONS_DIR": "torch_extensions_dir",
}

CACHE_FALLBACK_SUBDIRS = {
    "HF_HOME": "huggingface",
    "HF_DATASETS_CACHE": Path("huggingface") / "datasets",
    "TORCH_HOME": "torch",
    "XDG_CACHE_HOME": "xdg",
    "UV_CACHE_DIR": "uv",
    "PIP_CACHE_DIR": "pip",
    "TRITON_CACHE_DIR": "triton",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
}


def _wsl_to_windows(value: str) -> str:
    """Translate ``/mnt/<drive>/...`` to ``<DRIVE>:\\...`` on native Windows."""
    if os.name != "nt":
        return value
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", value.replace("\\", "/"))
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"
    return value


def normalize_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a user-supplied path, translating WSL mount notation if needed."""
    text = os.path.expanduser(str(value)).strip().strip('"').strip("'")
    text = _wsl_to_windows(text)
    return Path(text).expanduser().resolve()


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal, dependency-free .env parser (KEY=VALUE, comments, blank lines)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def discover_project_root() -> Path:
    """Find the repository root from env or by walking up from cwd."""
    env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("QUERYDISTILL_ROOT")
    if env_root:
        return normalize_path(env_root)
    candidate = Path.cwd().resolve()
    for parent in (candidate, *candidate.parents):
        marker = parent / "pyproject.toml"
        if marker.exists() and _REPO_MARKER in marker.read_text(encoding="utf-8", errors="ignore"):
            return parent
    return candidate


def detect_cache_root() -> Path:
    """Pick the cache root.

    Preference order: explicit XDG-style env var override for the cache root is
    not defined, so we use ``QUERYDISTILL_CACHE_ROOT`` if present, otherwise the
    required ``D:\\LLMCache`` location when D: exists, otherwise a project-local
    cache directory (which is reported as a warning by hardware-doctor).
    """
    override = os.environ.get("QUERYDISTILL_CACHE_ROOT")
    if override:
        return normalize_path(override)
    if os.name == "nt":
        d_drive = Path("D:/")
        if d_drive.exists():
            return (d_drive / "LLMCache").resolve()
        return Path("LLMCache").resolve()
    mount = Path("/mnt/d")
    if mount.exists():
        return (mount / "LLMCache").resolve()
    return Path("LLMCache").resolve()


@dataclass
class Settings:
    """All path configuration used by the project."""

    project_root: Path
    data_dir: Path
    model_dir: Path
    checkpoint_dir: Path
    artifact_dir: Path
    run_dir: Path
    cache_root: Path
    hf_home: Path
    hf_datasets_cache: Path
    torch_home: Path
    xdg_cache_home: Path
    uv_cache_dir: Path
    pip_cache_dir: Path
    triton_cache_dir: Path
    torch_extensions_dir: Path
    student_model_id: str = "Qwen/Qwen3-0.6B-Base"
    teacher_model_id: str = "Qwen/Qwen3-4B"
    max_execution_ms: int = 3000
    max_rows: int = 1000
    env_values: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, dotenv_path: Path | None = None) -> Settings:
        project_root = discover_project_root()
        env_path = dotenv_path or (project_root / ".env")
        env_values = _load_dotenv(env_path)
        merged = {**os.environ, **env_values}

        def path_of(key: str, default: Path) -> Path:
            value = merged.get(key)
            return normalize_path(value) if value else default.resolve()

        data_dir = path_of("DATA_DIR", project_root / "data")
        model_dir = path_of("MODEL_DIR", project_root / "models")
        checkpoint_dir = path_of("CHECKPOINT_DIR", project_root / "checkpoints")
        artifact_dir = path_of("ARTIFACT_DIR", project_root / "artifacts")
        run_dir = path_of("RUN_DIR", project_root / "runs")

        cache_root = (
            normalize_path(merged["QUERYDISTILL_CACHE_ROOT"])
            if merged.get("QUERYDISTILL_CACHE_ROOT")
            else detect_cache_root()
        )

        cache_paths: dict[str, Path] = {}
        for env_name, attr_name in CACHE_ENV_VARS.items():
            value = merged.get(env_name)
            if value:
                cache_paths[attr_name] = normalize_path(value)
            else:
                fallback = CACHE_FALLBACK_SUBDIRS[env_name]
                cache_paths[attr_name] = (cache_root / fallback).resolve()

        max_exec = int(merged.get("QD_MAX_EXECUTION_MS", "3000"))
        max_rows = int(merged.get("QD_MAX_ROWS", "1000"))

        return cls(
            project_root=project_root,
            data_dir=data_dir,
            model_dir=model_dir,
            checkpoint_dir=checkpoint_dir,
            artifact_dir=artifact_dir,
            run_dir=run_dir,
            cache_root=cache_root,
            **cache_paths,
            student_model_id=merged.get("STUDENT_MODEL_ID", "Qwen/Qwen3-0.6B-Base"),
            teacher_model_id=merged.get("TEACHER_MODEL_ID", "Qwen/Qwen3-4B"),
            max_execution_ms=max_exec,
            max_rows=max_rows,
            env_values=env_values,
        )

    def ensure_directories(self) -> list[Path]:
        created: list[Path] = []
        for path in (
            self.data_dir,
            self.model_dir,
            self.checkpoint_dir,
            self.artifact_dir,
            self.run_dir,
            self.cache_root,
            self.hf_home,
            self.hf_datasets_cache,
            self.torch_home,
            self.xdg_cache_home,
            self.uv_cache_dir,
            self.pip_cache_dir,
            self.triton_cache_dir,
            self.torch_extensions_dir,
            self.cache_root / "tmp",
        ):
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        return created

    def child_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Environment for subprocesses only. Never touches the user's environment."""
        env = dict(os.environ)
        # Explicit .env keys (e.g. HF_ENDPOINT) are forwarded to child
        # processes so model downloads and trainers honor them.
        env.update(self.env_values)
        for env_name, attr_name in CACHE_ENV_VARS.items():
            env[env_name] = str(getattr(self, attr_name))
        env["PROJECT_ROOT"] = str(self.project_root)
        env["DATA_DIR"] = str(self.data_dir)
        env["MODEL_DIR"] = str(self.model_dir)
        env["CHECKPOINT_DIR"] = str(self.checkpoint_dir)
        env["ARTIFACT_DIR"] = str(self.artifact_dir)
        env["RUN_DIR"] = str(self.run_dir)
        env["QD_MAX_EXECUTION_MS"] = str(self.max_execution_ms)
        env["QD_MAX_ROWS"] = str(self.max_rows)
        tmp_root = str(self.cache_root / "tmp")
        env["TMPDIR"] = tmp_root
        env["TMP"] = tmp_root
        env["TEMP"] = tmp_root
        if extra:
            env.update(extra)
        return env

    def smoke_dir(self, name: str) -> Path:
        return self.artifact_dir / "smoke" / name


def drive_of(path: Path) -> str:
    """Return the drive/mount the path lives on ('D:', '/mnt/d', 'other')."""
    text = str(path).replace("\\", "/")
    if os.name == "nt":
        if re.match(r"^[a-zA-Z]:/", text):
            return f"{text[0].upper()}:"
        return "unknown"
    match = re.match(r"^(/mnt/[a-zA-Z])/", text)
    if match:
        return match.group(1)
    return "unknown"


@dataclass
class CacheAuditItem:
    level: str  # "ok" | "warning" | "error"
    name: str
    detail: str


def audit_cache_paths(settings: Settings) -> list[CacheAuditItem]:
    """Check that every cache path avoids the small C: drive."""
    items: list[CacheAuditItem] = []
    project_drive = drive_of(settings.project_root)
    cache_drive = drive_of(settings.cache_root)

    if os.name == "nt":
        if project_drive == "C:":
            items.append(
                CacheAuditItem("error", "PROJECT_ROOT", "project root resolves to the C: drive")
            )
        else:
            items.append(CacheAuditItem("ok", "PROJECT_ROOT", f"project root on {project_drive}"))
        for env_name, attr_name in CACHE_ENV_VARS.items():
            path = getattr(settings, attr_name)
            if drive_of(path) == "C:":
                items.append(
                    CacheAuditItem(
                        "warning", env_name, f"{path} resolves to C: (space-constrained drive)"
                    )
                )
        if cache_drive == "C:":
            items.append(CacheAuditItem("error", "cache root", "cache root resolves to C:"))
        else:
            items.append(
                CacheAuditItem(
                    "ok", "cache root", f"cache root on {cache_drive} ({settings.cache_root})"
                )
            )
    else:
        wanted = "/mnt/d"
        if cache_drive == wanted:
            items.append(CacheAuditItem("ok", "cache root", f"cache root on {wanted}"))
        else:
            items.append(
                CacheAuditItem(
                    "warning", "cache root", f"cache root not on {wanted}: {settings.cache_root}"
                )
            )

    for name, path in (
        ("project root", settings.project_root),
        ("cache root", settings.cache_root),
    ):
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        free_gb = usage.free / (1024**3)
        threshold = 30 if name == "cache root" else 10
        if free_gb < threshold:
            items.append(
                CacheAuditItem(
                    "warning",
                    f"{name} free space",
                    f"only {free_gb:.1f} GiB free (threshold {threshold} GiB)",
                )
            )
    return items
