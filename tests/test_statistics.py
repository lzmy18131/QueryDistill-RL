"""Statistical test implementations (exact McNemar + paired bootstrap)."""

from __future__ import annotations

import pytest

from querydistill.evaluation.statistics import mcnemar_exact, paired_bootstrap_ci


def test_mcnemar_symmetry_and_ties():
    tied = mcnemar_exact([True, False], [False, True])
    assert tied["b"] == 1 and tied["c"] == 1
    assert tied["p_value"] == pytest.approx(1.0)
    assert tied["method"] == "exact binomial (McNemar)"


def test_mcnemar_hand_computed_value():
    # b=1, c=9: p = 2 * (P(X<=1)) for Binomial(10, 0.5) = 2 * 11/1024.
    model_a = [True] + [False] * 9
    model_b = [False] + [True] * 9
    result = mcnemar_exact(model_a, model_b)
    assert result["b"] == 1
    assert result["c"] == 9
    assert result["p_value"] == pytest.approx(22 / 1024)


def test_mcnemar_extreme_discordance_is_significant():
    result = mcnemar_exact([True] * 10, [False] * 10)
    assert result["p_value"] < 0.01
    assert result["discordant"] == 10


def test_mcnemar_requires_equal_length():
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])


def test_bootstrap_constant_difference_contains_zero():
    result = paired_bootstrap_ci([0.0] * 20, n_bootstrap=500, seed=42)
    assert result["mean_difference"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    assert result["n_pairs"] == 20


def test_bootstrap_positive_difference():
    result = paired_bootstrap_ci([0.1] * 30, n_bootstrap=1000, seed=7)
    assert result["mean_difference"] > 0.0
    assert result["ci_low"] > 0.0
    assert result["ci_high"] >= result["ci_low"]


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.0], n_bootstrap=10)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.0], alpha=1.5)


def test_bootstrap_empty_input_returns_zero():
    result = paired_bootstrap_ci([])
    assert result == {
        "mean_difference": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "n_bootstrap": 2000,
        "alpha": 0.05,
        "n_pairs": 0,
    }
