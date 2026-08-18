"""Prompt construction for policy models.

The prompt contains the natural-language question, optional evidence, and the
database schema. It NEVER contains gold SQL or gold execution results (see
LeakageGuard for the enforced invariant and ``tests/test_leakage_guard.py``).

Student protocol lock:
- ``student_prompt_version`` identifies the prompt content version.
- ``student_chat_template`` is the official Qwen/ChatML template used by the
  student tokenizer for GRPO and evaluation.
- SFT uses the same raw ``build_prompt`` user content; LLaMA-Factory applies the
  same underlying chat template during serialization.
"""

from __future__ import annotations

from .parser import ParseResult

student_prompt_version = "bird-v1"
student_chat_template = "qwen_chatml"

PROTOCOL_SPEC = """Answer in the following format only.

<plan>
tables: ...
joins: ...
filters: ...
grouping: ...
ordering: ...
</plan>
<sql>
SELECT ...
</sql>"""

PROTOCOL_SPEC_V2 = """Target dialect is SQLite.

Return exactly ONE read-only SQL statement.
Only SELECT or WITH ... SELECT.

Return exactly the columns requested by the question.
Do not add extra result/debug columns.
Use the supplied Evidence as authoritative schema/value guidance.
Do not invent tables or columns.

For SQLite date handling:
avoid EXTRACT().
Use SQLite-compatible expressions such as
strftime() / substr() when appropriate.

Never output multiple SQL statements.

Output exactly:

<sql>
...
</sql>"""


def prompt_protocol_spec(include_plan: bool = True, prompt_version: str = "bird-v1") -> str:
    if prompt_version == "bird-v2":
        return PROTOCOL_SPEC_V2
    if include_plan:
        return PROTOCOL_SPEC
    return """Answer in the following format only.

<sql>
SELECT ...
</sql>"""


def build_prompt(
    question: str,
    schema_text: str,
    db_id: str,
    include_plan: bool = True,
    dialect_note: str = "SQLite",
    evidence: str = "",
    prompt_version: str = "bird-v1",
) -> str:
    """Build the raw user content for one policy example.

    The return value is the exact user message text used by SFT, GRPO, and
    evaluation before chat-template serialization.
    """
    header = (
        f"Translate the following question into one {dialect_note} SQL query for "
        f"the database '{db_id}'."
    )
    parts = [header, f"Database schema:\n{schema_text}", f"Question: {question}"]
    if evidence:
        parts.append(f"Evidence: {evidence}")
    parts.append(prompt_protocol_spec(include_plan, prompt_version=prompt_version))
    return "\n\n".join(parts)


def build_student_messages(
    question: str,
    schema_text: str,
    db_id: str,
    include_plan: bool = True,
    evidence: str = "",
) -> list[dict[str, str]]:
    """Return the frozen student message list for an example."""
    return [
        {
            "role": "user",
            "content": build_prompt(
                question=question,
                schema_text=schema_text,
                db_id=db_id,
                include_plan=include_plan,
                evidence=evidence,
            ),
        }
    ]


def apply_student_chat_template(
    tokenizer,
    question: str,
    schema_text: str,
    db_id: str,
    include_plan: bool = True,
    evidence: str = "",
    add_generation_prompt: bool = True,
    tokenize: bool = False,
):
    """Serialize the frozen student prompt with the tokenizer chat template."""
    messages = build_student_messages(
        question=question,
        schema_text=schema_text,
        db_id=db_id,
        include_plan=include_plan,
        evidence=evidence,
    )
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
    )


def completion_to_sql(completion_text: str, require_plan: bool = False) -> ParseResult:
    """Small adapter used by the GRPO reward wrapper."""
    from .parser import parse_model_output

    return parse_model_output(completion_text, require_plan=require_plan)
