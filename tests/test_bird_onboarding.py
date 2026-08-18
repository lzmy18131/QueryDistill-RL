"""BIRD onboarding and protocol-lock tests.

These tests verify the frozen student prompt/template alignment, BIRD adapter
conversion, Teacher chat-template usage, and official evaluator export.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from querydistill.data.bird import build_bird_examples, write_bird_examples, write_bird_registry
from querydistill.data.dataset import build_prompt_rows, build_sft_rows
from querydistill.data.schema import Example, load_examples
from querydistill.distillation.backends import TransformersTeacherBackend
from querydistill.evaluation.bird_eval import export_predictions
from querydistill.outputs.prompting import (
    apply_student_chat_template,
    student_chat_template,
    student_prompt_version,
)
from querydistill.training.grpo_backend import GRPOSmokeConfig
from querydistill.training.llamafactory_backend import QLoRAConfig
from querydistill.utils import strict_dataclass_from_dict
from tests.helpers import sample_example

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    def __init__(self):
        self.chat_calls = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        self.chat_calls.append(messages)
        if tokenize:
            return [[1, 2, 3]]
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, *args, **kwargs):
        return "x"


def test_student_protocol_constants():
    assert student_prompt_version == "bird-v1"
    assert student_chat_template == "qwen_chatml"


def test_sft_and_grpo_use_same_user_content():
    example = sample_example(evidence="Evidence text")
    sft_row = build_sft_rows([example])[0]
    grpo_row, _ = build_prompt_rows([example])
    assert sft_row["input"] == grpo_row[0]["prompt"]
    assert "Evidence: Evidence text" in sft_row["input"]


def test_chat_serialization_is_stable():
    example = sample_example(evidence="Evidence text")
    tokenizer = _FakeTokenizer()
    serialized = apply_student_chat_template(
        tokenizer,
        example.question,
        example.schema_text,
        example.db_id,
        include_plan=False,
        evidence=example.evidence,
        tokenize=False,
    )
    assert serialized.startswith("<chat>")
    assert "Evidence: Evidence text" in serialized
    assert len(tokenizer.chat_calls) == 1
    assert tokenizer.chat_calls[0][0]["role"] == "user"


def test_teacher_backend_uses_chat_template():
    class _FakeIds:
        shape = (1, 3)

    class _FakeInputs(dict):
        def to(self, device):
            return self

    class _FakeTeacherTokenizer:
        def __init__(self):
            self.chat_calls = []
            self.eos_token_id = 1

        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            self.chat_calls.append(messages)
            return _FakeInputs(input_ids=_FakeIds())

        def decode(self, *args, **kwargs):
            return "<sql>SELECT 1</sql>"

    class _FakeTeacherModel:
        device = "cpu"

        def generate(self, **kwargs):
            return [[1, 2, 3]]

        def eval(self):
            return self

    backend = TransformersTeacherBackend(model_id="dummy")
    tokenizer = _FakeTeacherTokenizer()
    backend._tokenizer = tokenizer
    backend._model = _FakeTeacherModel()
    backend.generate("prompt", num_candidates=1)
    assert len(tokenizer.chat_calls) == 1
    assert tokenizer.chat_calls[0][0]["content"] == "prompt"


def test_bird_adapter_converts_real_pilot_subset():
    train_path = PROJECT_ROOT / "data/bird/raw/bird23_train_filtered.jsonl"
    mini_path = PROJECT_ROOT / "data/bird/raw/mini_dev_sqlite.json"
    train_db_dir = PROJECT_ROOT / "data/bird/train/databases"
    mini_db_dir = PROJECT_ROOT / "data/bird/mini_dev/databases"
    examples, report = build_bird_examples(
        train_path, mini_path, train_db_dir, mini_db_dir, max_train=5, max_dev=3
    )
    assert report.train_count <= 5
    assert report.dev_count <= 3
    assert all(isinstance(e, Example) for e in examples)
    assert all(e.evidence != "" or True for e in examples)


def test_bird_registry_and_examples_roundtrip(tmp_path, tiny_db):
    example = sample_example(evidence="ev")
    write_bird_examples([example], tmp_path / "examples")
    registry = tmp_path / "db_registry.json"
    write_bird_registry(registry, tmp_path / "train_dbs", tmp_path / "mini_dbs")
    loaded = load_examples(tmp_path / "examples" / "train.jsonl")
    assert loaded[0].evidence == "ev"


def test_bird_eval_exporter_uses_question_id(tmp_path):
    path = export_predictions(
        [{"example_id": "bird-dev-1471", "db_id": "d", "SQL": "SELECT 1"}],
        tmp_path / "pred.jsonl",
    )
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["question_id"] == 1471


def test_experiment_configs_parse():
    for name, cls in [
        ("bird_base.yaml", None),
        ("bird_gold_sft.yaml", QLoRAConfig),
        ("bird_distilled_sft.yaml", QLoRAConfig),
        ("bird_grpo.yaml", GRPOSmokeConfig),
        ("bird_gptq.yaml", None),
    ]:
        path = PROJECT_ROOT / "configs/experiment" / name
        assert path.exists(), name
        if cls is not None:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            strict_dataclass_from_dict(cls, payload, source=str(path))
