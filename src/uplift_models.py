"""Heterogeneous treatment effect (uplift) modeling: who does advertising actually
move, versus who converts (or doesn't) regardless of treatment.

Two models, in increasing sophistication:
  - T-learner: two independent classifiers (treated-only, control-only) predicting
    conversion probability; uplift = p(convert | treated) - p(convert | control).
    Simple, but effects for the two models are estimated independently, which can be
    noisy when the treatment effect is small relative to the outcome's base rate.
  - Causal forest (EconML's CausalForestDML): a single doubly-robust ensemble model
    that estimates the conditional average treatment effect (CATE) directly, which
    is generally better calibrated for heterogeneous-effect estimation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklift.metrics import qini_auc_score


@dataclass
class UpliftFit:
    model_name: str
    uplift_scores: np.ndarray
    qini_auc: float


def fit_t_learner(
    df: pd.DataFrame,
    features: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    random_state: int = 42,
) -> UpliftFit:
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    model_treated = HistGradientBoostingClassifier(random_state=random_state)
    model_control = HistGradientBoostingClassifier(random_state=random_state)
    model_treated.fit(treated[features], treated[outcome_col])
    model_control.fit(control[features], control[outcome_col])

    p1 = model_treated.predict_proba(df[features])[:, 1]
    p0 = model_control.predict_proba(df[features])[:, 1]
    uplift_scores = p1 - p0

    qini = qini_auc_score(df[outcome_col].values, uplift_scores, df[treatment_col].values)
    return UpliftFit(model_name="T-learner", uplift_scores=uplift_scores, qini_auc=qini)


def fit_causal_forest(
    df: pd.DataFrame,
    features: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    n_estimators: int = 200,
    random_state: int = 42,
) -> UpliftFit:
    cf = CausalForestDML(
        model_y=HistGradientBoostingClassifier(random_state=random_state),
        model_t=LogisticRegressionCV(cv=3, max_iter=1000),
        discrete_outcome=True,
        discrete_treatment=True,
        n_estimators=n_estimators,
        random_state=random_state,
        cv=2,
    )
    X = df[features].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    cf.fit(Y, T, X=X)
    uplift_scores = np.asarray(cf.effect(X)).ravel()

    qini = qini_auc_score(Y, uplift_scores, T)
    return UpliftFit(model_name="Causal Forest", uplift_scores=uplift_scores, qini_auc=qini)


def segment_uplift_table(
    df: pd.DataFrame,
    uplift_scores: np.ndarray,
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    n_bins: int = 5,
) -> pd.DataFrame:
    """Bucket users into DISJOINT uplift quintiles (Q5 = highest predicted uplift,
    Q1 = lowest) and report each segment's own observed conversion rates and lift --
    the table a marketer would use to decide who to target.

    Deliberately disjoint rather than sklift's cumulative top-X% convention: a
    cumulative "top 100%" bucket always just reproduces the overall average, which
    reads poorly as "the worst segment" even though it's the lowest cumulative value.
    """
    # Rank first (unique, tie-broken) rather than qcut on raw scores directly: with a
    # rare outcome, many predicted uplift values tie exactly, which can make qcut drop
    # bin edges and desync from a fixed-length label list. Ranking guarantees n_bins
    # equal-sized groups regardless of ties in the underlying scores.
    rank = pd.Series(uplift_scores).rank(method="first")
    segment = pd.qcut(rank, q=n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)])

    # Build from positional numpy arrays rather than assigning into df[...] by index:
    # uplift_scores (and therefore `segment`) is positionally aligned to df's rows, but
    # df often has a non-contiguous index (e.g. from .sample()), so an index-aligned
    # assignment would silently scramble the mapping.
    work = pd.DataFrame(
        {
            treatment_col: df[treatment_col].to_numpy(),
            outcome_col: df[outcome_col].to_numpy(),
            "segment": segment.to_numpy(),
        }
    )

    rows = []
    for seg, group in work.groupby("segment", observed=True):
        treated = group[group[treatment_col] == 1][outcome_col]
        control = group[group[treatment_col] == 0][outcome_col]
        conv_treated = treated.mean() if len(treated) else float("nan")
        conv_control = control.mean() if len(control) else float("nan")
        rows.append(
            {
                "segment": seg,
                "n_treatment": len(treated),
                "n_control": len(control),
                "conv_rate_if_treated": conv_treated,
                "conv_rate_if_not_treated": conv_control,
                "estimated_incremental_lift": conv_treated - conv_control,
            }
        )
    table = pd.DataFrame(rows).set_index("segment")
    return table.sort_index(ascending=False)  # Q5 (highest predicted uplift) first
