"""Config system tests: path resolution, D-drive rules, dotenv parsing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from querydistill.config import (
    _load_dotenv,
    _wsl_to_windows,
    audit_cache_paths,
    discover_project_root,
    normalize_path,
)
from tests.helpers import make_settings


def test_discover_project_root_from_pyproject():
    root = discover_project_root()
    assert (root / "pyproject.toml").exists()
    assert "querydistill" in (root / "pyproject.toml").read_text(encoding="utf-8")


def test_dotenv_parser_handles_comments_and_quotes(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "KEY_A=value-a\n# comment\nKEY_B = \"quoted value\"\nKEY_C='single'\nNOVALUE\n",
        encoding="utf-8",
    )
    values = _load_dotenv(env)
    assert values == {"KEY_A": "value-a", "KEY_B": "quoted value", "KEY_C": "single"}


def test_wsl_path_translation_windows():
    if os.name == "nt":
        assert _wsl_to_windows("/mnt/d/LLMCache/huggingface") == "D:\\LLMCache\\huggingface"
    else:
        assert _wsl_to_windows("/mnt/d/LLMCache") == "/mnt/d/LLMCache"


def test_settings_child_env_sets_cache_vars(tmp_path):
    settings = make_settings(tmp_path)
    env = settings.child_env()
    assert env["HF_HOME"] == str(settings.hf_home)
    assert env["HF_DATASETS_CACHE"] == str(settings.hf_datasets_cache)
    assert env["PROJECT_ROOT"] == str(settings.project_root)
    assert "QD_MAX_EXECUTION_MS" in env


def test_ensure_directories_creates_all(tmp_path):
    settings = make_settings(tmp_path)
    created = settings.ensure_directories()
    assert len(created) >= 14
    assert all(path.exists() for path in created)


def test_cache_audit_flags_c_drive(tmp_path, monkeypatch):
    if os.name != "nt":
        pytest.skip("C-drive audit is Windows-specific")
    settings = make_settings(tmp_path)
    settings.hf_home = Path("C:/Users/someone/.cache/huggingface")
    settings.cache_root = Path("C:/Users/someone/.cache")
    report = audit_cache_paths(settings)
    kinds = {(item.name, item.level) for item in report}
    assert any(level != "ok" for _, level in kinds)
    assert any("C:" in item.detail for item in report if item.level != "ok")


def test_cache_audit_ok_for_d_drive(tmp_path):
    if os.name == "nt" and not Path("D:/").exists():
        pytest.skip("D: not present")
    settings = make_settings(tmp_path)
    if os.name == "nt":
        settings.cache_root = Path("D:/LLMCache-test")
        settings.hf_home = Path("D:/LLMCache-test/huggingface")
        settings.project_root = Path("D:/LLMProjects-test/QueryDistill-RL")
    report = audit_cache_paths(settings)
    errors = [item for item in report if item.level == "error"]
    assert errors == []


def test_normalize_path_expands_user():
    path = normalize_path("~/somewhere")
    assert str(path) != "~/somewhere"
