"""Model output protocol parser tests (missing/duplicate tags, fences, prose...)."""

from __future__ import annotations

from querydistill.outputs.parser import parse_model_output

VALID = "<plan>\ntables: users\n</plan>\n<sql>\nSELECT * FROM users\n</sql>\n"
FENCED = "```sql\nSELECT 1\n```"


def test_valid_tags_extract_sql_and_plan():
    result = parse_model_output(VALID)
    assert result.format_ok
    assert result.sql == "SELECT * FROM users"
    assert result.plan and "users" in result.plan
    assert result.parse_error is None


def test_missing_tags_falls_back_to_single_fence():
    result = parse_model_output(FENCED)
    assert result.sql == "SELECT 1"
    assert not result.format_ok
    assert result.protocol == "fence"
    assert "missing <sql> tags" in (result.parse_error or "")


def test_duplicate_sql_tags_are_rejected():
    text = "<sql>SELECT 1</sql>\n<sql>SELECT 2</sql>"
    result = parse_model_output(text)
    assert result.sql is None
    assert not result.format_ok
    assert "duplicate" in (result.parse_error or "")


def test_two_sql_tags_never_execute_second():
    text = "<sql>SELECT name FROM users</sql>\n<sql>DROP TABLE users</sql>"
    result = parse_model_output(text)
    assert result.sql is None
    assert "duplicate" in (result.parse_error or "")


def test_markdown_fence_inside_tags_is_stripped():
    text = "<sql>\n```sql\nSELECT 1\n```\n</sql>"
    result = parse_model_output(text)
    assert result.sql == "SELECT 1"
    assert result.format_ok
    assert any("fence" in warning for warning in result.warnings)


def test_whitespace_is_stripped():
    result = parse_model_output("<sql>\n   SELECT  1 ;  \n</sql>")
    assert result.sql == "SELECT  1 ;"


def test_extra_prose_outside_tags_is_ignored():
    text = "Sure, here it is:\n<sql>SELECT 1</sql>\nHope this helps!"
    result = parse_model_output(text)
    assert result.sql == "SELECT 1"
    assert result.format_ok
    assert any("extra prose" in warning for warning in result.warnings)


def test_unclosed_tag_is_malformed():
    result = parse_model_output("<sql>SELECT 1")
    assert result.sql is None
    assert not result.format_ok
    assert "unbalanced" in (result.parse_error or "")


def test_mismatched_close_tag_is_malformed():
    result = parse_model_output("<sql>SELECT 1</plan>")
    assert result.sql is None
    assert not result.format_ok


def test_empty_sql_block_is_error():
    result = parse_model_output("<sql>\n   \n</sql>")
    assert result.sql is None
    assert "empty" in (result.parse_error or "")


def test_require_plan_rejects_missing_plan_but_keeps_sql():
    result = parse_model_output("<sql>SELECT 1</sql>", require_plan=True)
    assert result.sql == "SELECT 1"
    assert not result.format_ok
    assert "<plan>" in (result.parse_error or "")


def test_no_tags_no_fence_is_error():
    result = parse_model_output("I cannot answer this.")
    assert result.sql is None
    assert result.protocol == "none"


def test_multiple_fences_are_rejected():
    result = parse_model_output("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```")
    assert result.sql is None
    assert "multiple" in (result.parse_error or "")


def test_case_insensitive_tags():
    result = parse_model_output("<SQL>\nSELECT 1\n</SQL>")
    assert result.sql == "SELECT 1"
