# SQL safety design (P0)

Model-generated SQL is untrusted code. Defense in depth:

## Layer 1 - sqlglot AST validation

`src/querydistill/sql/safety.py` allows exactly one top-level statement of type
SELECT (including `WITH ... SELECT`, subqueries, UNION of selects). Rejected:
INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, ATTACH, DETACH, VACUUM,
any PRAGMA, TRIGGER, `load_extension`, MERGE, and multiple statements
(including `SELECT 1; DROP TABLE x`). Trailing semicolons and comments do not
count as extra statements.

## Layer 2 - SQLite read-only + authorizer + process isolation

`src/querydistill/sql/executor.py`:

1. Database opens with `file:...?mode=ro` and `PRAGMA query_only=ON`.
2. Authorizer allows only `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`
   (denying `load_extension`) and `SQLITE_RECURSIVE`; everything else is
   `SQLITE_DENY` (fail closed).
3. Progress handler raises `TimeoutError` after `max_execution_ms`.
4. Watchdog thread calls `connection.interrupt()` at the deadline.
5. The worker runs in a **separate spawned process**; the parent joins with a
   hard deadline and `terminate()`s a stuck worker. A thread-only timeout is
   explicitly not relied upon.
6. `max_rows` bounds rows returned to the caller (truncation is reported).

## Environment contract

* The model supplies only `db_id` (validated against `^[A-Za-z0-9][A-Za-z0-9_-]*$`).
* `SQLExecutionEnvironment` resolves db_id through an explicit registry
  file to an allowlisted absolute path; filesystem paths from the model are
  impossible by construction.
* Every registry entry must resolve inside the registry root.

## Tested attack surface

DROP / DELETE / UPDATE / INSERT / ATTACH / DETACH / write-PRAGMA / VACUUM /
CREATE / multiple statements / semicolon tricks / comments / WITH-SELECT /
nested SELECT / UNION SELECT / recursive CTE timeout / expensive Cartesian
product / `load_extension`. See `tests/test_sql_safety.py`,
`tests/test_sql_executor.py`, `tests/test_reward_hacking.py`.
