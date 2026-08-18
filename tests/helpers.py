"""Shared test helpers (no fixtures here; importable from test modules)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from querydistill.config import Settings
from querydistill.data.schema import Example


def make_settings(tmp_path: Path) -> Settings:
    root = tmp_path / "project"
    cache = tmp_path / "cache"
    return Settings(
        project_root=root,
        data_dir=root / "data",
        model_dir=root / "models",
        checkpoint_dir=root / "checkpoints",
        artifact_dir=root / "artifacts",
        run_dir=root / "runs",
        cache_root=cache,
        hf_home=cache / "huggingface",
        hf_datasets_cache=cache / "huggingface" / "datasets",
        torch_home=cache / "torch",
        xdg_cache_home=cache / "xdg",
        uv_cache_dir=cache / "uv",
        pip_cache_dir=cache / "pip",
        triton_cache_dir=cache / "triton",
        torch_extensions_dir=cache / "torch_extensions",
    )


def build_tiny_db(path: Path) -> Path:
    """users + orders tables used by safety / reward tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)"
        )
        connection.executemany(
            "INSERT INTO users VALUES (?, ?, ?)",
            [(1, "Alice", 30), (2, "Bob", 25), (3, "Carol", 35)],
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(1, 1, 10.5), (2, 1, 20.0), (3, 2, 5.25)],
        )
    connection.close()
    return path


def sample_example(**overrides) -> Example:
    payload = {
        "example_id": "ex-001",
        "db_id": "tiny",
        "question": "List user names.",
        "schema_text": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);",
        "gold_sql": "SELECT name FROM users",
        "split": "train",
        "source": "test",
        "source_version": "1.0",
    }
    payload.update(overrides)
    return Example.model_validate(payload)
