import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportions_ztest

from src.causal_estimates import compute_itt_lift


def _synthetic_df(n_treatment, conversions_treatment, n_control, conversions_control, seed=0):
    rng = np.random.default_rng(seed)
    treatment = np.concatenate([np.ones(n_treatment), np.zeros(n_control)])
    conversion = np.zeros(n_treatment + n_control)

    treated_idx = rng.choice(n_treatment, size=conversions_treatment, replace=False)
    conversion[treated_idx] = 1

    control_idx = n_treatment + rng.choice(n_control, size=conversions_control, replace=False)
    conversion[control_idx] = 1

    return pd.DataFrame({"treatment": treatment, "conversion": conversion})


def test_absolute_and_relative_lift_match_hand_computed_values():
    df = _synthetic_df(n_treatment=10_000, conversions_treatment=1200, n_control=10_000, conversions_control=1000)
    result = compute_itt_lift(df)

    assert result.conv_rate_treatment == pytest.approx(0.12)
    assert result.conv_rate_control == pytest.approx(0.10)
    assert result.absolute_lift == pytest.approx(0.02)
    assert result.relative_lift == pytest.approx(0.20)
    assert result.incremental_conversions == pytest.approx(0.02 * 10_000)


def test_z_stat_and_p_value_match_independent_statsmodels_implementation():
    df = _synthetic_df(n_treatment=10_000, conversions_treatment=1200, n_control=10_000, conversions_control=1000)
    result = compute_itt_lift(df)

    expected_z, expected_p = proportions_ztest(
        count=[1200, 1000], nobs=[10_000, 10_000]
    )
    assert result.z_stat == pytest.approx(expected_z, rel=1e-6)
    assert result.p_value == pytest.approx(expected_p, rel=1e-6)


def test_large_clear_effect_is_significant():
    df = _synthetic_df(n_treatment=10_000, conversions_treatment=1200, n_control=10_000, conversions_control=1000)
    result = compute_itt_lift(df)

    assert result.significant_at_05 is True
    assert result.ci_low > 0  # CI excludes zero
    assert result.logit_p_value < 0.05


def test_small_noisy_effect_is_not_significant():
    df = _synthetic_df(n_treatment=1_000, conversions_treatment=120, n_control=1_000, conversions_control=100)
    result = compute_itt_lift(df)

    assert result.significant_at_05 is False
    assert result.ci_low < 0 < result.ci_high  # CI straddles zero


def test_zero_lift_gives_near_zero_estimate():
    df = _synthetic_df(n_treatment=5_000, conversions_treatment=500, n_control=5_000, conversions_control=500)
    result = compute_itt_lift(df)

    assert result.absolute_lift == pytest.approx(0.0, abs=1e-9)
    assert result.significant_at_05 is False


def test_confidence_interval_contains_point_estimate():
    df = _synthetic_df(n_treatment=10_000, conversions_treatment=1200, n_control=10_000, conversions_control=1000)
    result = compute_itt_lift(df)

    assert result.ci_low < result.absolute_lift < result.ci_high
