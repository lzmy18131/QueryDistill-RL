"""QueryDistill-RL command line interface."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Annotated

import typer
import yaml

from . import __version__
from .config import Settings
from .utils import atomic_write_json, strict_dataclass_from_dict, utc_now_iso

app = typer.Typer(
    name="querydistill",
    help="QueryDistill-RL: Text-to-SQL small-LLM post-training and compression.",
    no_args_is_help=True,
)


def _settings() -> Settings:
    return Settings.load()


def _resolve(settings: Settings, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = settings.project_root / candidate
    return candidate.resolve()


def _record_status(settings: Settings, name: str, payload: dict) -> Path:
    target = settings.smoke_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": utc_now_iso(), **payload}
    atomic_write_json(target / "status.json", payload)
    return target / "status.json"


@app.command("hardware-doctor")
def hardware_doctor(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit a JSON report instead of the text table.")
    ] = False,
) -> None:
    """Print OS / CPU / RAM / GPU / CUDA / dependency versions and D-drive cache audit."""
    from .hardware import format_report, probe

    report = probe(_settings())
    if json_output:
        typer.echo(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        typer.echo(format_report(report))


@app.command("version")
def version() -> None:
    """Print the package version."""
    typer.echo(f"querydistill {__version__}")


@app.command("make-fixtures")
def make_fixtures(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing fixture files.")
    ] = False,
    database_dir: Annotated[
        str | None, typer.Option("--database-dir", help="Output directory for the .db files.")
    ] = None,
    examples_path: Annotated[
        str | None, typer.Option("--examples-path", help="Output JSONL for examples.")
    ] = None,
) -> None:
    """Generate the 3 tiny SQLite databases and 42 synthetic examples."""
    settings = _settings()
    settings.ensure_directories()
    fixture_root = _resolve(settings, database_dir or "tests/fixtures/tiny_sql")
    database_dir_path = fixture_root / "databases"
    examples = _resolve(settings, examples_path or "data/tiny_sql/examples.jsonl")
    registry = fixture_root / "db_registry.json"
    manifest = fixture_root / "fixtures_manifest.json"

    from .data.fixtures import make_fixtures as _make

    result = _make(
        database_dir=database_dir_path,
        examples_path=examples,
        registry_path=registry,
        manifest_path=manifest,
        force=force,
    )
    typer.echo(
        f"fixtures written: {result.example_count} examples "
        f"(splits={result.split_counts}), databases={sorted(result.sha256_by_db)}"
    )
    typer.echo(f"registry: {registry}")
    typer.echo(f"examples: {examples}")


@app.command("audit-data")
def audit_data(
    examples_path: Annotated[
        str | None, typer.Option("--examples-path", help="Examples JSONL to audit.")
    ] = None,
    registry_path: Annotated[
        str | None, typer.Option("--registry-path", help="db_id -> path registry JSON.")
    ] = None,
    distillation_path: Annotated[
        str | None, typer.Option("--distillation-path", help="Optional distillation JSONL.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Audit example schema, fixture DBs, leakage and gold execution sanity."""
    settings = _settings()
    examples = _resolve(settings, examples_path or "data/tiny_sql/examples.jsonl")
    registry = _resolve(settings, registry_path or "tests/fixtures/tiny_sql/db_registry.json")
    distillation = _resolve(settings, distillation_path) if distillation_path else None
    from .data.audit import audit_data as _audit
    from .data.audit import format_audit

    report = _audit(examples, registry, distillation)
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        typer.echo(format_audit(report))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("prepare-bird")
def prepare_bird(
    train_path: Annotated[str | None, typer.Option("--train-path")] = None,
    mini_dev_path: Annotated[str | None, typer.Option("--mini-dev-path")] = None,
    train_db_dir: Annotated[str | None, typer.Option("--train-db-dir")] = None,
    mini_dev_db_dir: Annotated[str | None, typer.Option("--mini-dev-db-dir")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir")] = None,
    registry_path: Annotated[str | None, typer.Option("--registry-path")] = None,
    max_train: Annotated[int | None, typer.Option("--max-train")] = 50,
    max_dev: Annotated[int | None, typer.Option("--max-dev")] = 20,
) -> None:
    """Prepare a real BIRD pilot subset into project Example JSONL + SQLite registry."""
    settings = _settings()
    train_file = _resolve(settings, train_path or "data/bird/raw/bird23_train_filtered.jsonl")
    mini_file = _resolve(settings, mini_dev_path or "data/bird/raw/mini_dev_sqlite.json")
    train_dbs = _resolve(settings, train_db_dir or "data/bird/train/databases")
    mini_dbs = _resolve(settings, mini_dev_db_dir or "data/bird/mini_dev/databases")
    out_dir = _resolve(settings, output_dir or "data/bird/examples")
    reg_path = _resolve(settings, registry_path or "data/bird/db_registry.json")

    from .data.bird import build_bird_examples, write_bird_examples, write_bird_registry

    examples, report = build_bird_examples(
        train_file,
        mini_file,
        train_dbs,
        mini_dbs,
        max_train=max_train,
        max_dev=max_dev,
    )
    write_bird_examples(examples, out_dir)
    write_bird_registry(reg_path, train_dbs, mini_dbs)
    payload = report.as_dict()
    payload["output_dir"] = str(out_dir)
    payload["registry_path"] = str(reg_path)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command("evaluate")
def evaluate(
    backend: Annotated[str, typer.Option("--backend", help="mock | transformers")] = "mock",
    split: Annotated[str | None, typer.Option("--split", help="dev | test (required)")] = None,
    mock_strategy: Annotated[
        str, typer.Option("--mock-strategy", help="gold | wrong | unsafe | malformed | timeout")
    ] = "gold",
    model_path: Annotated[str | None, typer.Option("--model-path")] = None,
    model_spec: Annotated[
        str | None,
        typer.Option("--model-spec", help="YAML ModelSpec for --backend transformers"),
    ] = None,
    stage: Annotated[
        str | None,
        typer.Option("--stage", help="base | adapter | merged | gptq"),
    ] = None,
    base_model: Annotated[str | None, typer.Option("--base-model")] = None,
    adapter: Annotated[str | None, typer.Option("--adapter")] = None,
    examples_path: Annotated[str | None, typer.Option("--examples-path")] = None,
    registry_path: Annotated[str | None, typer.Option("--registry-path")] = None,
    max_examples: Annotated[int | None, typer.Option("--max-examples")] = None,
    plan: Annotated[bool, typer.Option("--plan", help="Require the <plan> section.")] = False,
    output: Annotated[str | None, typer.Option("--output", help="Write JSON report here.")] = None,
) -> None:
    """Run the unified evaluation harness against an explicit dev/test split."""
    settings = _settings()
    examples_file = _resolve(settings, examples_path or "data/tiny_sql/examples.jsonl")
    registry = _resolve(settings, registry_path or "tests/fixtures/tiny_sql/db_registry.json")

    from .data.schema import load_examples
    from .data.split_policy import require_explicit_eval_split
    from .evaluation.harness import EvaluationHarness, MockModelBackend, TransformersModelBackend
    from .sql.environment import SQLExecutionEnvironment

    split = require_explicit_eval_split(split)

    environment = SQLExecutionEnvironment.from_registry(
        registry, max_rows=settings.max_rows, max_execution_ms=settings.max_execution_ms
    )
    examples = load_examples(examples_file)
    spec = None
    if backend == "mock":
        oracle = {example.example_id: example.gold_sql for example in examples}
        model_backend = MockModelBackend(strategy=mock_strategy, gold_oracle=oracle)
    elif backend == "transformers":
        from .evaluation.modelspec import ModelSpec, infer_model_spec, load_model

        if model_spec:
            spec_payload = yaml.safe_load(
                _resolve(settings, model_spec).read_text(encoding="utf-8")
            )
            spec = strict_dataclass_from_dict(ModelSpec, spec_payload, source=str(model_spec))
        elif stage:
            if stage not in {"base", "adapter", "merged", "gptq"}:
                raise typer.BadParameter(f"unsupported --stage {stage!r}")
            spec = ModelSpec(
                stage=stage,
                base_model_path=str(_resolve(settings, base_model)) if base_model else None,
                adapter_path=str(_resolve(settings, adapter)) if adapter else None,
                merged_model_path=str(_resolve(settings, model_path))
                if stage == "merged"
                else None,
                quantized_model_path=str(_resolve(settings, model_path))
                if stage == "gptq"
                else None,
            )
        else:
            if not model_path:
                raise typer.BadParameter(
                    "provide --model-path (base/merged/gptq), --model-spec, "
                    "or --stage + --base-model/--adapter/--model-path"
                )
            spec = infer_model_spec(_resolve(settings, model_path), stage="base")

        model, tokenizer = load_model(spec)
        model_backend = TransformersModelBackend(model_path=spec.primary_path() or str(model_path))
        model_backend._model = model
        model_backend._tokenizer = tokenizer
    else:
        raise typer.BadParameter(f"unknown backend {backend!r}")

    harness = EvaluationHarness(environment, model_backend, require_plan=plan)
    metrics = harness.run(examples, split=split, max_examples=max_examples)
    report = {
        "backend": model_backend.name,
        "metrics": metrics.aggregate(),
        "records": [record.as_dict() for record in metrics.records],
    }
    if output:
        target = _resolve(settings, output)
        atomic_write_json(target, report)
        from .utils import utc_now_iso

        identity = {
            "backend": model_backend.name,
            "split": split,
            "created_at": utc_now_iso(),
        }
        if spec is not None:
            identity.update(spec.identity())
        elif model_path:
            identity["model_path"] = str(_resolve(settings, model_path))
        identity_path = target.parent / f"{target.stem}.model_identity.json"
        atomic_write_json(identity_path, identity)
        typer.echo(f"evaluation report written to {target}")
        typer.echo(f"model identity written to {identity_path}")
    typer.echo(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


distill_app = typer.Typer(help="Teacher distillation candidate generation.", no_args_is_help=True)
app.add_typer(distill_app, name="distill")


@distill_app.command("generate")
def distill_generate(
    backend: Annotated[str, typer.Option("--backend", help="mock | transformers")] = "mock",
    mock_strategy: Annotated[str, typer.Option("--mock-strategy", help="gold | constant")] = "gold",
    examples_path: Annotated[str | None, typer.Option("--examples-path")] = None,
    registry_path: Annotated[str | None, typer.Option("--registry-path")] = None,
    output_path: Annotated[str | None, typer.Option("--output")] = None,
    num_candidates: Annotated[int, typer.Option("--num-candidates")] = 1,
    max_samples: Annotated[int | None, typer.Option("--max-samples")] = None,
    max_attempts: Annotated[int, typer.Option("--max-attempts")] = 1,
    seed: Annotated[int, typer.Option("--seed")] = 3407,
    target_verified_examples: Annotated[
        int | None, typer.Option("--target-verified-examples")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue from existing output.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    teacher_config_path: Annotated[
        str | None,
        typer.Option("--teacher-config", help="YAML TeacherConfig for --backend transformers"),
    ] = None,
    teacher_model: Annotated[str | None, typer.Option("--teacher-model")] = None,
    teacher_revision: Annotated[str | None, typer.Option("--teacher-revision")] = None,
    teacher_prompt_version: Annotated[str | None, typer.Option("--teacher-prompt-version")] = None,
) -> None:
    """Generate verified teacher distillation candidates (mock or real, resumable)."""
    settings = _settings()
    examples = _resolve(settings, examples_path or "data/tiny_sql/examples.jsonl")
    registry = _resolve(settings, registry_path or "tests/fixtures/tiny_sql/db_registry.json")
    output = _resolve(settings, output_path or "data/distilled/mock_smoke.jsonl")

    from .distillation.backends import TeacherConfig
    from .distillation.pipeline import DistillationConfig, DistillationPipeline

    if backend == "mock":
        teacher_id = "mock-teacher-1.0"
        teacher_rev = "unknown"
        prompt_version = "v1"
        generation_config = {"note": "test-only mock teacher; never reported as a real teacher"}
        backend_kwargs = {"strategy": mock_strategy}
    elif backend == "transformers":
        if teacher_config_path:
            teacher_payload = yaml.safe_load(
                _resolve(settings, teacher_config_path).read_text(encoding="utf-8")
            )
            teacher_cfg = strict_dataclass_from_dict(
                TeacherConfig, teacher_payload, source=str(teacher_config_path)
            )
        else:
            teacher_cfg = TeacherConfig(
                model_id=teacher_model or settings.teacher_model_id,
                revision=teacher_revision or "main",
                prompt_version=teacher_prompt_version or "v1",
            )
        teacher_id = teacher_cfg.model_id
        teacher_rev = teacher_cfg.revision
        prompt_version = teacher_cfg.prompt_version
        generation_config = teacher_cfg.generation_config()
        backend_kwargs = teacher_cfg.backend_kwargs()
    else:
        raise typer.BadParameter(f"unknown backend {backend!r}")

    config = DistillationConfig(
        examples_path=examples,
        registry_path=registry,
        output_path=output,
        teacher_model=teacher_id,
        teacher_model_revision=teacher_rev,
        teacher_prompt_version=prompt_version,
        num_candidates=num_candidates,
        max_samples=max_samples,
        resume=resume,
        dry_run=dry_run,
        backend_name=backend,
        backend_kwargs=backend_kwargs,
        generation_config=generation_config,
        max_attempts=max_attempts,
        seed=seed,
        target_verified_examples=target_verified_examples,
    )
    pipeline = DistillationPipeline(config)
    result = pipeline.run()
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("train-sft")
def train_sft(
    config_path: Annotated[str, typer.Option("--config", help="QLoRAConfig YAML file.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    no_status: Annotated[
        bool, typer.Option("--no-status", help="Do not write smoke status.")
    ] = False,
) -> None:
    """Run LLaMA-Factory QLoRA SFT with a project-generated YAML."""
    settings = _settings()
    config_file = _resolve(settings, config_path)
    from .training.llamafactory_backend import (
        LLaMAFactoryNotInstalledError,
        LLaMAFactoryRunError,
        QLoRAConfig,
        run_llamafactory,
    )

    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = strict_dataclass_from_dict(QLoRAConfig, payload, source=str(config_file))
    config.output_dir = str(_resolve(settings, config.output_dir))
    config.logging_dir = str(_resolve(settings, config.logging_dir))
    config.dataset_dir = str(_resolve(settings, config.dataset_dir))

    # Build the project's own train-only SFT dataset before invoking LLaMA-Factory.
    from .artifacts.manifest import ArtifactManifest, config_hash, hash_or_none
    from .data.dataset import build_sft_rows
    from .data.paired import build_paired_targets
    from .data.schema import load_distillation_records, load_examples
    from .data.split_policy import SplitPolicy, assert_training_splits
    from .training.llamafactory_backend import prepare_dataset_dir

    examples_path = _resolve(settings, config.source_examples_path)
    assert_training_splits(config.train_splits, policy_name="sft")
    policy = SplitPolicy(allowed_splits=set(config.train_splits), policy_name="train_only")
    examples, split_report = policy.apply(load_examples(examples_path), source_path=examples_path)

    from .utils import load_json

    paired_manifest_path = (
        _resolve(settings, config.paired_manifest_path)
        if config.paired_manifest_path
        else Path(config.dataset_dir) / "paired_manifest.json"
    )
    if config.target_mode == "distilled":
        if not config.distillation_path:
            raise typer.BadParameter(
                "target_mode=distilled requires distillation_path in the config"
            )
        records = load_distillation_records(_resolve(settings, config.distillation_path))
        paired = build_paired_targets(
            examples,
            records,
            examples_path=examples_path,
            require_all=config.strict_distilled,
            selection_policy="min_candidate_index",
        )
        target_sql_by_id = paired.distilled_targets
        examples = [e for e in examples if e.example_id in paired.distilled_targets]
        paired.write_manifest(paired_manifest_path)
    elif config.target_mode == "gold":
        if config.paired_manifest_path:
            if not paired_manifest_path.exists():
                raise typer.BadParameter(
                    f"paired_manifest_path does not exist: {paired_manifest_path}"
                )
            manifest = load_json(paired_manifest_path)
            paired_ids = set(manifest["example_ids"])
            examples = [e for e in examples if e.example_id in paired_ids]
        target_sql_by_id = None
    else:
        raise typer.BadParameter(f"unknown target_mode {config.target_mode!r}")

    rows = build_sft_rows(
        examples, target_sql_by_id=target_sql_by_id, include_plan=config.include_plan
    )
    prepare_dataset_dir(rows, Path(config.dataset_dir), alias=config.dataset_alias)
    split_report_path = Path(config.dataset_dir) / "split_report.json"
    atomic_write_json(split_report_path, split_report.as_dict())

    run_id = uuid.uuid4().hex
    run_dir = settings.artifact_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = run_dir / "llamafactory_config.yaml"
    log_path = run_dir / "trainer.log"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    resolved_yaml = yaml.safe_dump(config.resolved_dict(), sort_keys=False, allow_unicode=True)
    (run_dir / "resolved_config.yaml").write_text(resolved_yaml, encoding="utf-8")
    try:
        report = run_llamafactory(
            config,
            yaml_path,
            log_path,
            dry_run=dry_run,
            env=settings.child_env(),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    except (LLaMAFactoryNotInstalledError, LLaMAFactoryRunError) as exc:
        if not no_status:
            _record_status(
                settings,
                "sft",
                {"status": "BLOCKED", "reason": str(exc), "dry_run": dry_run, "trained": False},
            )
        typer.echo(f"BLOCKED: {exc}")
        raise typer.Exit(code=0) from None
    if not no_status:
        _record_status(
            settings,
            "sft",
            {"status": "PASS" if not dry_run else "DRY_RUN", **report, "trained": not dry_run},
        )
    if not dry_run:
        manifest = ArtifactManifest(
            stage="sft",
            input_artifact=str(examples_path),
            input_sha256=hash_or_none(examples_path),
            output_artifact=str(Path(config.output_dir)),
            base_model=str(Path(config.model_name_or_path).resolve())
            if Path(config.model_name_or_path).exists()
            else config.model_name_or_path,
            adapter=str(Path(config.output_dir)),
            config_hash=config_hash(config),
            extra={
                "split_report": split_report.as_dict(),
                "target_mode": config.target_mode,
                "paired_manifest": str(paired_manifest_path),
                "run_id": run_id,
            },
        )
        manifest.write(config.output_dir)
        manifest.write(run_dir)
        # Copy the adapter manifest into the run directory as well.
        import shutil

        src_adapter = Path(config.output_dir) / "adapter_model.safetensors"
        if src_adapter.exists():
            shutil.copy2(src_adapter, run_dir / "adapter_model.safetensors")
        metrics = {
            "run_id": run_id,
            "target_mode": config.target_mode,
            "paired_manifest": str(paired_manifest_path),
            "train_row_count": len(rows),
            "steps": config.max_steps,
            "peak_vram_mib": None,
            "duration_seconds": None,
            "adapter_sha256": None,
        }
        atomic_write_json(run_dir / "metrics.json", metrics)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("train-grpo")
def train_grpo(
    config_path: Annotated[str, typer.Option("--config", help="GRPOSmokeConfig YAML file.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    no_status: Annotated[bool, typer.Option("--no-status")] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue an identical existing GRPO run.")
    ] = False,
) -> None:
    """Run TRL GRPO with the real SQLite reward environment (smoke only)."""
    settings = _settings()
    config_file = _resolve(settings, config_path)
    from .training.grpo_backend import GRPOBlockedError, GRPOSmokeConfig, GRPOSmokeRunner

    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = strict_dataclass_from_dict(GRPOSmokeConfig, payload, source=str(config_file))
    config.dry_run = dry_run
    config.resume = resume
    config.base_model_path = str(_resolve(settings, config.base_model_path))
    if config.init_adapter_path:
        config.init_adapter_path = str(_resolve(settings, config.init_adapter_path))
    if config.init_merged_model_path:
        config.init_merged_model_path = str(_resolve(settings, config.init_merged_model_path))
    config.examples_path = str(_resolve(settings, config.examples_path))
    config.registry_path = str(_resolve(settings, config.registry_path))
    config.output_dir = str(_resolve(settings, config.output_dir))

    runner = GRPOSmokeRunner(config)
    try:
        result = runner.run(dry_run=dry_run)
    except GRPOBlockedError as exc:
        typer.echo(f"BLOCKED: {exc}")
        typer.echo(f"status file: {settings.smoke_dir('grpo') / 'status.json'}")
        return
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not no_status and not dry_run:
        _record_status(settings, "grpo", result)


@app.command("quantize-gptq")
def quantize_gptq(
    config_path: Annotated[str, typer.Option("--config", help="GPTQConfig YAML file.")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Merge LoRA (optional) and quantize to GPTQ INT4 via GPTQModel."""
    settings = _settings()
    config_file = _resolve(settings, config_path)
    from .quantization.gptq import GPTQConfig, GPTQRunner

    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = strict_dataclass_from_dict(GPTQConfig, payload, source=str(config_file))
    config.dry_run = dry_run
    config.base_model_path = str(_resolve(settings, config.base_model_path))
    if config.adapter_path:
        config.adapter_path = str(_resolve(settings, config.adapter_path))
    if config.merged_output_dir:
        config.merged_output_dir = str(_resolve(settings, config.merged_output_dir))
    config.output_dir = str(_resolve(settings, config.output_dir))
    config.calibration_examples_path = str(_resolve(settings, config.calibration_examples_path))

    runner = GPTQRunner(config)
    result = runner.run(dry_run=dry_run)
    _record_status(settings, "gptq", result)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("serve-vllm")
def serve_vllm(
    config_path: Annotated[str, typer.Option("--config", help="VLLMServeConfig YAML file.")],
    print_command: Annotated[
        bool, typer.Option("--print-command", help="Only print the vllm serve command.")
    ] = False,
) -> None:
    """Validate vLLM serving configuration and optionally print the launch command."""
    settings = _settings()
    config_file = _resolve(settings, config_path)
    from .serving.vllm import VLLMServeConfig, build_server_command, check_compatibility

    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config = strict_dataclass_from_dict(VLLMServeConfig, payload, source=str(config_file))
    config.model_path = str(_resolve(settings, config.model_path))

    problems = config.validate()
    compatibility = (
        check_compatibility(config.model_path) if Path(config.model_path).exists() else None
    )
    command = build_server_command(config)
    status = {
        "status": "DRY_RUN" if (problems or print_command) else "NOT_STARTED",
        "config_problems": problems,
        "command": command,
        "compatibility": compatibility,
        "started": False,
    }
    _record_status(settings, "vllm", status)
    typer.echo(json.dumps(status, ensure_ascii=False, indent=2))
    if print_command:
        typer.echo(" ".join(command))


@app.command("benchmark")
def benchmark(
    prompt: Annotated[str | None, typer.Option("--prompt", help="Single prompt text.")] = None,
    prompts_file: Annotated[
        str | None, typer.Option("--prompts-file", help="JSONL file with 'prompt' fields.")
    ] = None,
    endpoint: Annotated[str, typer.Option("--endpoint")] = "http://127.0.0.1:8000/v1",
    model: Annotated[str, typer.Option("--model", help="Served model name.")] = "querydistill",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 1,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 128,
    output: Annotated[str | None, typer.Option("--output")] = None,
    api_key: Annotated[
        str | None, typer.Option("--api-key", help="vLLM API key (or VLLM_API_KEY env).")
    ] = None,
) -> None:
    """Benchmark an OpenAI-compatible endpoint (TTFT / latency / tokens / VRAM)."""
    settings = _settings()
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt)
    if prompts_file:
        from .utils import load_jsonl

        for record in load_jsonl(_resolve(settings, prompts_file)):
            prompts.append(str(record["prompt"]))
    if not prompts:
        raise typer.BadParameter("provide --prompt or --prompts-file")

    from .serving.vllm import benchmark as _benchmark
    from .serving.vllm import save_benchmark

    report = _benchmark(
        prompts=prompts,
        endpoint=endpoint,
        model=model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        api_key=api_key or os.environ.get("VLLM_API_KEY"),
    )
    if output:
        save_benchmark(report, _resolve(settings, output))
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("report")
def report(
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Aggregate artifact smoke statuses into reports/STATUS.md."""
    from .reporting.report import build_report, format_report

    built = build_report(_settings())
    if json_output:
        typer.echo(json.dumps(built, ensure_ascii=False, indent=2))
    else:
        typer.echo(format_report(built))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
