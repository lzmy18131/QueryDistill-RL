"""Execution reward: did the SQL run successfully inside the sandbox?"""

from __future__ import annotations

from ..sql.executor import ExecutionResult

SUCCESS = 0.1
FAILURE = 0.0


def execution_reward(result: ExecutionResult | None) -> tuple[float, dict]:
    if result is None:
        return 0.0, {"reason": "not executed (unsafe or unparseable)"}
    if result.success:
        return SUCCESS, {"reason": "executed without error", "row_count": result.row_count}
    return FAILURE, {
        "reason": "execution failed",
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
