# Result equivalence rules

Correctness is judged on **execution results**, never on SQL string equality.
`generated_sql == gold_sql` is not the primary correctness signal.

## Normalization

* Rows are compared as tuples of SQLite-native values.
* `NULL` equals `NULL`; `NULL` never equals the string `"NULL"`.
* `int` and `float` compare with relative tolerance (default `1e-6`), so
  `1` and `1.0000001` are equivalent but `1.0` and `1.01` are not.
  (Multiset canonicalization uses decimal quantization at the tolerance.)
* Strings are compared exactly and case-sensitively.
* Duplicate rows matter: the comparison is a **multiset**, so
  `[a, a, b] != [a, b, b]`.
* Column count must match even for empty results.

## Ordering

* Gold has no semantically relevant `ORDER BY`: multiset comparison.
* Gold has `ORDER BY`: ordered comparison; candidate row order must match.
  (`has_order_by` is detected on the gold AST.)

## Empty-result rule (anti reward hacking)

If both gold and candidate produce zero rows, the verifier does **not**
immediately return equivalent. It requires:

1. candidate SQL parses as a SELECT with a FROM source,
2. candidate tables are a subset of gold tables,
3. (when schema tables are supplied) candidate references only known tables.

If sanity passes, the verdict is `empty_structural` and the correctness reward
is capped at 0.25 instead of 1.0. This means `SELECT ... WHERE 1=0` can never
be the main path to a high reward.

## Failure rules

* Gold execution failure -> cannot verify.
* Candidate execution failure -> not equivalent (reason includes the error).
* Verification output carries `kind`, `reason`, row counts, column match and
  order sensitivity so every decision is auditable.
