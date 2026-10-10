"""Illustrative cost/revenue assumptions for turning a causal lift estimate into a
business decision (incremental ROAS, incremental cost per conversion).

The Criteo dataset has no cost or revenue fields at all -- these constants are
EXAMPLE values only, chosen to be plausible for a display-ad campaign, so the
iROAS/CPA math below is reproducible and clearly labeled as an assumption rather
than a fact about the dataset. Swap them for real campaign numbers before using
this for an actual budget decision.
"""

from dataclasses import dataclass

COST_PER_IMPRESSION = 0.01  # USD, i.e. a $10 CPM
REVENUE_PER_CONVERSION = 45.0  # USD, average order value assumption


@dataclass
class BudgetImpact:
    n_treated: int
    total_cost: float
    incremental_conversions: float
    incremental_revenue: float
    incremental_cost_per_conversion: float
    incremental_roas: float


def compute_budget_impact(
    n_treated: int,
    incremental_conversions: float,
    cost_per_impression: float = COST_PER_IMPRESSION,
    revenue_per_conversion: float = REVENUE_PER_CONVERSION,
) -> BudgetImpact:
    """Translate an incremental-conversions estimate into cost/revenue terms.

    Assumes one impression per treated user (the Criteo dataset doesn't record
    impression counts), so `n_treated * cost_per_impression` approximates spend.
    """
    total_cost = n_treated * cost_per_impression
    incremental_revenue = incremental_conversions * revenue_per_conversion
    incremental_cost_per_conversion = (
        total_cost / incremental_conversions if incremental_conversions > 0 else float("inf")
    )
    incremental_roas = incremental_revenue / total_cost if total_cost > 0 else float("nan")

    return BudgetImpact(
        n_treated=n_treated,
        total_cost=total_cost,
        incremental_conversions=incremental_conversions,
        incremental_revenue=incremental_revenue,
        incremental_cost_per_conversion=incremental_cost_per_conversion,
        incremental_roas=incremental_roas,
    )
