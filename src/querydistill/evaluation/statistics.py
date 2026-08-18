"""Statistical tests for model comparisons (implemented, not yet run on big data).

* McNemar exact test (two-sided, exact binomial) for paired binary outcomes
* paired bootstrap confidence interval for the accuracy difference

These must only be used with real experimental results; the tiny synthetic
fixture is never a basis for statistical claims (enforced by documentation).
"""

from __future__ import annotations


def mcnemar_exact(model_a_correct: list[bool], model_b_correct: list[bool]) -> dict:
    """Two-sided exact McNemar test on paired predictions.

    Returns discordant counts b/c (A right / B wrong and vice versa), total n,
    two-sided exact p-value and the midpoint p for ties.
    """
    if len(model_a_correct) != len(model_b_correct):
        raise ValueError("paired prediction lists must have equal length")
    b = sum(1 for a, c in zip(model_a_correct, model_b_correct, strict=True) if a and not c)
    c = sum(1 for a, c in zip(model_a_correct, model_b_correct, strict=True) if not a and c)
    discordant = b + c
    p_value = _two_sided_binomial(min(b, c), discordant)
    return {
        "b": b,
        "c": c,
        "discordant": discordant,
        "n": len(model_a_correct),
        "p_value": p_value,
        "two_sided": True,
        "method": "exact binomial (McNemar)",
    }


def _two_sided_binomial(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    probability = 0.5**n
    cumulative = 0.0
    combination = 1.0
    for i in range(k + 1):
        if i > 0:
            combination = combination * (n - i + 1) / i
        cumulative += combination * probability
    return min(1.0, 2.0 * cumulative)


def paired_bootstrap_ci(
    differences: list[float],
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for the mean of paired score differences."""
    if not differences:
        return {
            "mean_difference": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
            "n_pairs": 0,
        }
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    import numpy as np

    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_difference": float(values.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "n_pairs": len(values),
        "method": "paired bootstrap (BCa not applied; percentile method)",
    }
