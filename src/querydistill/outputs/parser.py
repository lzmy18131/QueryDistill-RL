"""Robust parser for the unified model output protocol.

Expected format (plan section configurable)::

    <plan>
    tables: ...
    joins: ...
    filters: ...
    grouping: ...
    ordering: ...
    </plan>
    <sql>
    SELECT ...
    </sql>

Only the single SQL statement inside ``<sql>...</sql>`` is ever executed.
The parser is deliberately strict about duplicate/unbalanced tags because a
hidden second SQL block is treated as a potential injection attempt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TAG_OPEN_RE = re.compile(r"<\s*(sql|plan)\b[^>]*>", re.IGNORECASE)
_TAG_CLOSE_RE = re.compile(r"<\s*/\s*(sql|plan)\s*>", re.IGNORECASE)
_FENCE_RE = re.compile(
    r"(?m)^[ \t]*```(?:sql)?[ \t]*\n(.*?)^[ \t]*```[ \t]*$", re.IGNORECASE | re.DOTALL
)


@dataclass
class ParseResult:
    sql: str | None
    plan: str | None
    raw: str
    format_ok: bool
    parse_error: str | None = None
    warnings: list[str] = field(default_factory=list)
    protocol: str = "tags"

    def as_dict(self) -> dict:
        return {
            "sql": self.sql,
            "plan": self.plan,
            "format_ok": self.format_ok,
            "parse_error": self.parse_error,
            "warnings": self.warnings,
            "protocol": self.protocol,
        }


def _strip_outer_fence(content: str) -> tuple[str, bool]:
    """Remove one complete wrapping markdown code fence if present."""
    content = content.strip()
    if content.startswith("```"):
        match = re.match(r"^```(?:sql)?[ \t]*\n", content, re.IGNORECASE)
        if match and content.rstrip().endswith("```"):
            inner = content[match.end() :]
            if inner.rstrip().endswith("```"):
                inner = inner.rstrip()[:-3]
                return inner.strip(), True
    return content, False


def parse_model_output(text: str, require_plan: bool = False) -> ParseResult:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    warnings: list[str] = []

    opens = [(m.group(1).lower(), m.start(), m.end()) for m in _TAG_OPEN_RE.finditer(raw)]
    closes = [(m.group(1).lower(), m.start(), m.end()) for m in _TAG_CLOSE_RE.finditer(raw)]

    sql_opens = [m for m in opens if m[0] == "sql"]
    sql_closes = [m for m in closes if m[0] == "sql"]
    plan_opens = [m for m in opens if m[0] == "plan"]
    plan_closes = [m for m in closes if m[0] == "plan"]

    def unbalanced(tag: str, open_count: int, close_count: int) -> str | None:
        if open_count == 0 and close_count == 0:
            return None
        if open_count != close_count:
            return f"unbalanced <{tag}> tags ({open_count} open, {close_count} close)"
        return None

    sql_unbalanced = unbalanced("sql", len(sql_opens), len(sql_closes))
    plan_unbalanced = unbalanced("plan", len(plan_opens), len(plan_closes))

    if sql_opens or sql_closes:
        if sql_unbalanced:
            return ParseResult(None, None, raw, False, sql_unbalanced, warnings, "tags")
        if plan_unbalanced:
            return ParseResult(None, None, raw, False, plan_unbalanced, warnings, "tags")
        if len(sql_opens) > 1:
            return ParseResult(
                None,
                None,
                raw,
                False,
                f"duplicate <sql> tags ({len(sql_opens)} found); refusing to execute",
                warnings,
                "tags",
            )

        sql_start = sql_opens[0][2]
        sql_end = sql_closes[0][1]
        if sql_end <= sql_start:
            return ParseResult(
                None, None, raw, False, "empty or inverted <sql> block", warnings, "tags"
            )

        sql_text, had_fence = _strip_outer_fence(raw[sql_start:sql_end])
        sql = sql_text.strip() if sql_text else None
        if sql is None:
            return ParseResult(None, None, raw, False, "<sql> block is empty", warnings, "tags")
        if had_fence:
            warnings.append("SQL block wrapped in a markdown fence")

        plan: str | None = None
        if plan_opens and plan_closes:
            if len(plan_opens) > 1:
                warnings.append("duplicate <plan> blocks; first block used")
            plan_start = plan_opens[0][2]
            plan_end = plan_closes[0][1]
            if plan_end > plan_start:
                plan = raw[plan_start:plan_end].strip()
        if require_plan and not plan:
            return ParseResult(
                sql,
                None,
                raw,
                False,
                "<plan> section is required but missing",
                warnings,
                "tags",
            )

        outside = (raw[: sql_opens[0][1]] + raw[sql_closes[0][2] :]).strip()
        if outside:
            warnings.append("extra prose outside <sql> tags is ignored")
        return ParseResult(sql, plan, raw, True, None, warnings, "tags")

    # No tags at all: fall back to exactly one markdown SQL fence.
    fences = _FENCE_RE.findall(raw)
    if len(fences) == 1:
        sql = fences[0].strip() or None
        return ParseResult(
            sql,
            None,
            raw,
            False,
            "missing <sql> tags (markdown fence fallback)",
            ["protocol uses a markdown fence instead of <sql> tags"],
            "fence",
        )
    if len(fences) > 1:
        return ParseResult(
            None,
            None,
            raw,
            False,
            "multiple markdown SQL fences; refusing to execute",
            warnings,
            "fence",
        )
    return ParseResult(
        None, None, raw, False, "no <sql> block and no markdown SQL fence found", warnings, "none"
    )
