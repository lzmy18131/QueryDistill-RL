#!/usr/bin/env python3
"""Package the repository for final pre-training review.

Creates QueryDistill-RL-final-pretraining-review.zip next to the project root.

Included: source, tests, configs, docs, small SQLite fixtures, small real
artifacts (json/jsonl/yaml/log/md), requirements, scripts.
Excluded: virtualenvs, model weights, HF cache, large checkpoints, large
datasets, bytecode, caches, the ZIP itself.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv-wsl",
    ".venv-sft",
    ".venv-serve",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "models",
    "checkpoints",
    "runs",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".zip",
    ".whl",
    ".tar",
    ".tar.gz",
    ".gz",
}


def _included(project_root: Path, path: Path) -> bool:
    relative = path.relative_to(project_root)
    parts = relative.parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return False
    if relative.name in {".env"}:
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    # Keep tiny smoke logs but not huge trainer checkpoints.
    return not path.stat().st_size > 50 * 1024 * 1024


def build_zip(project_root: Path, output: Path) -> Path:
    count = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for root, dirs, files in os.walk(project_root, topdown=True):
            # Prune excluded dirs BEFORE descending into them (venvs contain
            # WSL symlinks that Windows cannot stat).
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for name in sorted(files):
                path = Path(root) / name
                if output.resolve() == path.resolve():
                    continue
                if not _included(project_root, path):
                    continue
                try:
                    archive.write(
                        path,
                        str(Path("QueryDistill-RL") / path.relative_to(project_root)),
                    )
                except OSError:
                    continue
                count += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <project-root>/../QueryDistill-RL-final-pretraining-review.zip",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output = (
        Path(args.output)
        if args.output
        else project_root.parent / "QueryDistill-RL-final-pretraining-review.zip"
    )
    build_zip(project_root, output)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"wrote {output} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
