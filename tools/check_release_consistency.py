#!/usr/bin/env python3
"""Release consistency check for QueryDistill-RL.

Verifies that the tree that was tested is the same tree that was packaged:

1. scripts/ contains exactly the formal entry points.
2. PRETRAINING_GATE_REPORT.md exists and mentions the pytest gate.
3. For every source/config/docs file in the ZIP, its hash matches the project tree.
4. The ZIP is extracted to a temp dir and:
   - the extracted scripts/ set is exact,
   - compileall passes on src/tests/scripts,
   - a small set of structural/config tests passes from the extracted tree.

Usage:
  python scripts/check_release_consistency.py --project-root . --zip QueryDistill-RL-final-pretraining-review.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EXPECTED_SCRIPTS = {
    "benchmark_vllm.py",
    "gpu_smoke.sh",
    "package_release.py",
    "run_cpu_checks.sh",
    "run_inference_smoke.py",
    "serve_vllm.sh",
}

STRUCTURAL_TESTS = [
    "tests/test_round2_1.py::test_scripts_directory_is_clean",
    "tests/test_round2_2.py::test_scripts_directory_exact_set_again",
    "tests/test_round2_2.py::test_formal_distilled_local_has_strict_false",
    "tests/test_round2_2.py::test_grpo_resume_flag_does_not_change_training_fingerprint",
    "tests/test_round2_1.py::test_strict_config_loader_rejects_unknown_fields",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_scripts(project_root: Path) -> None:
    scripts = {p.name for p in (project_root / "scripts").glob("*") if p.is_file()}
    assert scripts == EXPECTED_SCRIPTS, f"scripts mismatch: {scripts}"


def check_report(project_root: Path) -> None:
    report = project_root / "docs" / "PRETRAINING_GATE_REPORT.md"
    assert report.exists(), "PRETRAINING_GATE_REPORT.md missing"
    text = report.read_text(encoding="utf-8")
    assert "pytest -q" in text, "report does not mention pytest gate"
    assert "READY_FOR_FULL_EXPERIMENT" in text, "report status not final"


def check_zip_hashes(project_root: Path, zip_path: Path) -> None:
    included_dirs = {"src", "tests", "configs", "docs", "scripts"}
    root_files = {"pyproject.toml", "README.md", "MODEL_CARD.md", "THIRD_PARTY.md"}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            rel = Path(info.filename)
            if len(rel.parts) < 2 or rel.parts[0] != "QueryDistill-RL":
                continue
            project_rel = Path(*rel.parts[1:])
            if project_rel.parts and project_rel.parts[0] in included_dirs:
                local = project_root / project_rel
                if not local.is_file():
                    raise AssertionError(f"zip contains file missing from project: {project_rel}")
                if sha256(local) != sha256(Path(local)):
                    # This branch is intentionally unreachable; keep hashes explicit below.
                    pass
                data = archive.read(info.filename)
                if hashlib.sha256(data).hexdigest() != sha256(local):
                    raise AssertionError(f"zip file differs from project: {project_rel}")
            elif project_rel.name in root_files:
                local = project_root / project_rel
                data = archive.read(info.filename)
                if hashlib.sha256(data).hexdigest() != sha256(local):
                    raise AssertionError(f"zip root file differs from project: {project_rel}")


def run_in_extracted(extracted: Path, python: str) -> None:
    scripts = {p.name for p in (extracted / "scripts").glob("*") if p.is_file()}
    assert scripts == EXPECTED_SCRIPTS, f"extracted scripts mismatch: {scripts}"
    subprocess.run(
        [python, "-m", "compileall", "-q", "src", "tests", "scripts"],
        cwd=extracted,
        check=True,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(extracted / "src")
    subprocess.run(
        [python, "-m", "pytest", "-q", *STRUCTURAL_TESTS],
        cwd=extracted,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--zip", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    zip_path = Path(args.zip).resolve()
    check_scripts(project_root)
    check_report(project_root)
    check_zip_hashes(project_root, zip_path)

    with tempfile.TemporaryDirectory(prefix="qd-release-check-") as tmp:
        extracted = Path(tmp) / "QueryDistill-RL"
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(Path(tmp))
        run_in_extracted(extracted, args.python)

    print("release consistency: OK")


if __name__ == "__main__":
    main()
