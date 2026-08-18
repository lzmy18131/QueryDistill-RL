"""Layer 1 of the SQL safety stack: sqlglot AST validation.

Model-generated SQL is untrusted code. Before it ever reaches SQLite we parse
it with sqlglot and reject:

* multiple statements (including the ``SELECT 1; DROP TABLE x`` trick)
* every mutating / administrative statement class
* ``PRAGMA``, ``VACUUM``, ``ATTACH``, ``DETACH``, ``TRIGGER``
* ``load_extension`` and the SQL ``REPLACE INTO`` spelling

The scalar function ``REPLACE(...)`` is read-only and is allowed inside a
single top-level SELECT / WITH-SELECT / UNION.  The destructive ``REPLACE
INTO`` form is parsed by sqlglot as a ``Command`` (or ``INSERT OR REPLACE`` as
an ``Insert``), so it is rejected by the statement and command checks below;
we do not ban the ``replace`` AST key itself because that key is also used by
the read-only scalar function.

Only a single top-level ``SELECT`` (or ``UNION`` of selects, or ``WITH ...
SELECT``) is allowed through to Layer 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

# sqlglot expression keys that are never allowed in the generated SQL.
FORBIDDEN_KEYS = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "vacuum",
        "command",
        "merge",
        "load",
        "reindex",
        "trigger",
        "transaction",
        "savepoint",
    }
)

FORBIDDEN_COMMAND_PREFIXES = (
    "ATTACH",
    "DETACH",
    "VACUUM",
    "PRAGMA",
    "CREATE",
    "DROP",
    "ALTER",
    "INSERT",
    "UPDATE",
    "DELETE",
    "REPLACE",
    "REINDEX",
    "LOAD",
)

FORBIDDEN_FUNCTIONS = frozenset({"load_extension"})


@dataclass
class SafetyDecision:
    safe: bool
    reason: str
    error_type: str  # "none" | "syntax_error" | "unsafe_sql"
    statement_count: int = 0
    top_level_key: str = ""
    forbidden_nodes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "safe": self.safe,
            "reason": self.reason,
            "error_type": self.error_type,
            "statement_count": self.statement_count,
            "top_level_key": self.top_level_key,
            "forbidden_nodes": self.forbidden_nodes,
        }


def _normalized_name(node: exp.Expression) -> str:
    name = node.name or getattr(node, "this", None)
    return str(name or "").strip().lower()


def _find_forbidden(node: exp.Expression, found: list[str]) -> None:
    key = getattr(node, "key", "") or ""
    text = str(node)
    upper = text.lstrip().upper()

    if isinstance(node, exp.Func) and _normalized_name(node) in FORBIDDEN_FUNCTIONS:
        found.append(f"function:{_normalized_name(node)}")
    if isinstance(node, exp.Command):
        found.append(f"command:{text.strip()[:60]}")
    elif key in FORBIDDEN_KEYS:
        found.append(f"{key}:{text.strip()[:60]}")
    if upper.startswith("REPLACE INTO"):
        found.append(f"replace:{text.strip()[:60]}")

    for child in node.iter_expressions():
        _find_forbidden(child, found)


def _is_select_like(expression: exp.Expression) -> bool:
    return isinstance(expression, (exp.Select, exp.Union))


def validate_sql(sql: str, dialect: str = "sqlite") -> SafetyDecision:
    """AST-validate one candidate SQL statement (Layer 1)."""
    text = (sql or "").strip()
    if not text:
        return SafetyDecision(
            safe=False, reason="empty SQL string", error_type="format_error", statement_count=0
        )

    try:
        statements = sqlglot.parse(text, read=dialect)
    except Exception as exc:  # sqlglot raises its own ParseError family
        return SafetyDecision(
            safe=False,
            reason=f"sqlglot parse error: {exc}",
            error_type="syntax_error",
            statement_count=-1,
        )

    if not statements:
        return SafetyDecision(
            safe=False,
            reason="no SQL statement found",
            error_type="syntax_error",
            statement_count=0,
        )
    # sqlglot emits a trailing Semicolon / None node for "SELECT 1;;" and
    # "SELECT 1; -- comment". Those carry no executable statement and must not
    # be mistaken for a multi-statement attack.
    statements = [
        statement
        for statement in statements
        if statement is not None
        and not isinstance(statement, exp.Semicolon)
        and str(statement).strip() not in {"", "None"}
    ]
    if not statements:
        return SafetyDecision(
            safe=False,
            reason="no SQL statement found",
            error_type="syntax_error",
            statement_count=0,
        )
    if len(statements) != 1:
        return SafetyDecision(
            safe=False,
            reason=f"multiple statements detected ({len(statements)})",
            error_type="unsafe_sql",
            statement_count=len(statements),
        )

    top = statements[0]
    if not _is_select_like(top):
        return SafetyDecision(
            safe=False,
            reason=f"top-level statement is {type(top).__name__}, not SELECT/WITH-SELECT/UNION",
            error_type="unsafe_sql",
            statement_count=1,
            top_level_key=getattr(top, "key", ""),
        )

    forbidden: list[str] = []
    _find_forbidden(top, forbidden)
    if forbidden:
        return SafetyDecision(
            safe=False,
            reason=f"forbidden AST nodes: {', '.join(forbidden[:5])}",
            error_type="unsafe_sql",
            statement_count=1,
            top_level_key=getattr(top, "key", ""),
            forbidden_nodes=forbidden,
        )

    return SafetyDecision(
        safe=True,
        reason="single read-only SELECT statement",
        error_type="none",
        statement_count=1,
        top_level_key=getattr(top, "key", ""),
    )
