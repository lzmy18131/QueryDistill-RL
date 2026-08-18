"""Shared pytest fixtures.

All test databases are created in tmp dirs; tests never depend on repo state
and never touch the real D:\\ cache directories.
"""

from __future__ import annotations

import pytest

from querydistill.sql.environment import SQLExecutionEnvironment
from tests.helpers import build_tiny_db, make_settings, sample_example  # noqa: F401


@pytest.fixture()
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture()
def tiny_db(tmp_path):
    return build_tiny_db(tmp_path / "tiny.db")


@pytest.fixture()
def tiny_environment(tiny_db):
    return SQLExecutionEnvironment({"tiny": tiny_db}, max_rows=100, max_execution_ms=1000)


@pytest.fixture()
def tiny_example(tiny_db):
    return sample_example(
        db_id="tiny",
        schema_text=(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);\n"
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);"
        ),
    )
