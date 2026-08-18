"""CLI smoke tests via typer CliRunner (no network, no models)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from querydistill.cli import app

runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_hardware_doctor_json():
    result = runner.invoke(app, ["hardware-doctor", "--json"])
    assert result.exit_code == 0
    assert '"torch_version"' in result.output
    assert '"cache_audit"' in result.output


def test_make_fixtures_and_audit(tmp_path):
    database_dir = tmp_path / "fixtures"
    examples = tmp_path / "examples.jsonl"
    created = runner.invoke(
        app,
        [
            "make-fixtures",
            "--database-dir",
            str(database_dir),
            "--examples-path",
            str(examples),
        ],
    )
    assert created.exit_code == 0, created.output
    registry = database_dir / "db_registry.json"
    assert registry.exists()
    assert (database_dir / "databases" / "shop.db").exists()

    audited = runner.invoke(
        app,
        ["audit-data", "--examples-path", str(examples), "--registry-path", str(registry)],
    )
    assert audited.exit_code == 0, audited.output
    assert "data audit ok: True" in audited.output


def test_evaluate_mock_gold_on_fixtures(tmp_path):
    database_dir = tmp_path / "fixtures"
    examples = tmp_path / "examples.jsonl"
    assert (
        runner.invoke(
            app,
            [
                "make-fixtures",
                "--database-dir",
                str(database_dir),
                "--examples-path",
                str(examples),
            ],
        ).exit_code
        == 0
    )
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--backend",
            "mock",
            "--mock-strategy",
            "gold",
            "--split",
            "test",
            "--examples-path",
            str(examples),
            "--registry-path",
            str(database_dir / "db_registry.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    # Round-2 strictness: empty structural results are partial credit, not
    # execution accuracy. The mock-gold CLI smoke must not claim 100% when the
    # tiny fixture includes intentionally empty-result test queries.
    assert metrics["execution_accuracy"] < 1.0
    assert metrics["partial_equivalence_rate"] > 0.0
    assert metrics["exact_match_secondary"] == 1.0


def test_evaluate_mock_gold_writes_model_identity(tmp_path):
    database_dir = tmp_path / "fixtures"
    examples = tmp_path / "examples.jsonl"
    assert (
        runner.invoke(
            app,
            [
                "make-fixtures",
                "--database-dir",
                str(database_dir),
                "--examples-path",
                str(examples),
            ],
        ).exit_code
        == 0
    )
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--backend",
            "mock",
            "--mock-strategy",
            "gold",
            "--split",
            "test",
            "--examples-path",
            str(examples),
            "--registry-path",
            str(database_dir / "db_registry.json"),
            "--output",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    identity_path = tmp_path / "report.model_identity.json"
    assert identity_path.exists()
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["backend"] == "mock"
    assert identity["split"] == "test"


def test_distill_generate_dry_run_and_run(tmp_path):
    database_dir = tmp_path / "fixtures"
    examples = tmp_path / "examples.jsonl"
    assert (
        runner.invoke(
            app,
            [
                "make-fixtures",
                "--database-dir",
                str(database_dir),
                "--examples-path",
                str(examples),
            ],
        ).exit_code
        == 0
    )
    output = tmp_path / "distilled.jsonl"
    dry = runner.invoke(
        app,
        [
            "distill",
            "generate",
            "--dry-run",
            "--max-samples",
            "3",
            "--examples-path",
            str(examples),
            "--registry-path",
            str(database_dir / "db_registry.json"),
            "--output",
            str(output),
        ],
    )
    assert dry.exit_code == 0, dry.output
    assert '"dry_run": true' in dry.output
    assert not output.exists()

    real = runner.invoke(
        app,
        [
            "distill",
            "generate",
            "--max-samples",
            "2",
            "--examples-path",
            str(examples),
            "--registry-path",
            str(database_dir / "db_registry.json"),
            "--output",
            str(output),
        ],
    )
    assert real.exit_code == 0, real.output
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_report_command_writes_status(tmp_path):
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
    assert "grpo" in result.output
