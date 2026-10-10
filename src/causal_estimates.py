"""Intention-to-treat (ITT) incrementality estimate: lift, confidence interval, significance.

Because treatment assignment in the Criteo dataset is randomized, the simple
difference in conversion rates between treatment and control is an unbiased
estimate of the causal effect of being assigned to see ads (the ITT effect).
No adjustment for confounders is needed here -- that is the whole point of a
randomized experiment, and is the estimate a marketer should trust over "conversions
attributed to the ad platform".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


@dataclass
class ITTResult:
    n_treatment: int
    n_control: int
    conv_rate_treatment: float
    conv_rate_control: float
    absolute_lift: float
    relative_lift: float
    incremental_conversions: float
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float
    significant_at_05: bool
    # cross-check via logistic regression with robust (HC1) standard errors
    logit_coef_treatment: float
    logit_p_value: float

    def summary(self) -> str:
        return (
            f"Treatment conversion rate: {self.conv_rate_treatment:.4%} (n={self.n_treatment:,})\n"
            f"Control conversion rate:   {self.conv_rate_control:.4%} (n={self.n_control:,})\n"
            f"Absolute lift: {self.absolute_lift:.4%}  (95% CI: [{self.ci_low:.4%}, {self.ci_high:.4%}])\n"
            f"Relative lift: {self.relative_lift:.2%}\n"
            f"Incremental conversions (in treatment group): {self.incremental_conversions:,.0f}\n"
            f"z = {self.z_stat:.2f}, p = {self.p_value:.2e} "
            f"({'significant' if self.significant_at_05 else 'NOT significant'} at alpha=0.05)\n"
            f"Logistic regression cross-check: treatment coef p-value = {self.logit_p_value:.2e}"
        )


def compute_itt_lift(df: pd.DataFrame, outcome_col: str = "conversion",
                      treatment_col: str = "treatment", alpha: float = 0.05) -> ITTResult:
    """Two-proportion z-test for the ITT effect of `treatment_col` on `outcome_col`.

    Also fits conversion ~ treatment via logistic regression with heteroskedasticity-
    robust (HC1) standard errors as an independent cross-check of significance.
    """
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]

    n1, n0 = len(treated), len(control)
    p1, p0 = treated.mean(), control.mean()

    absolute_lift = p1 - p0
    relative_lift = absolute_lift / p0 if p0 > 0 else np.nan
    incremental_conversions = absolute_lift * n1

    # pooled SE for the significance test, unpooled SE for the CI (standard two-proportion practice)
    p_pool = (treated.sum() + control.sum()) / (n1 + n0)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    z_stat = absolute_lift / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = absolute_lift - z_crit * se_unpooled
    ci_high = absolute_lift + z_crit * se_unpooled

    X = sm.add_constant(df[treatment_col].astype(float))
    y = df[outcome_col].astype(float)
    logit_model = sm.Logit(y, X).fit(disp=0, cov_type="HC1")
    logit_p_value = logit_model.pvalues[treatment_col]

    return ITTResult(
        n_treatment=n1,
        n_control=n0,
        conv_rate_treatment=p1,
        conv_rate_control=p0,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        incremental_conversions=incremental_conversions,
        ci_low=ci_low,
        ci_high=ci_high,
        z_stat=z_stat,
        p_value=p_value,
        significant_at_05=bool(p_value < alpha),
        logit_coef_treatment=logit_model.params[treatment_col],
        logit_p_value=logit_p_value,
    )
