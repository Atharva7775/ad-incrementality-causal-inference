"""Streamlit dashboard: incremental ad impact summary and budget-allocation
recommendation, built on top of the project's src/ causal-inference modules.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.causal_estimates import compute_itt_lift
from src.cost_assumptions import compute_budget_impact
from src.data_loader import FEATURE_COLS, load_sample
from src.uplift_models import fit_t_learner, segment_uplift_table

st.set_page_config(page_title="Ad Incrementality Dashboard", layout="wide")

st.title("Incremental Ad Impact & Budget Allocation")
st.caption(
    "Causal measurement of a display-ad campaign using the Criteo Uplift Prediction "
    "Dataset (randomized incrementality test). All cost/revenue figures below are "
    "illustrative assumptions, not provided by the dataset -- adjust them in the sidebar."
)


@st.cache_data
def get_data():
    return load_sample()


@st.cache_data
def get_itt_result(_df):
    return compute_itt_lift(_df)


@st.cache_data
def get_uplift_segments(_df):
    fit = fit_t_learner(_df, FEATURE_COLS)
    return segment_uplift_table(_df, fit.uplift_scores, n_bins=5)


df = get_data()
result = get_itt_result(df)

st.sidebar.header("Cost & revenue assumptions")
st.sidebar.caption("Not provided by the dataset -- illustrative, editable inputs.")
cost_per_impression = st.sidebar.number_input(
    "Cost per impression ($)", min_value=0.0, value=0.01, step=0.005, format="%.3f"
)
revenue_per_conversion = st.sidebar.number_input(
    "Revenue per conversion ($)", min_value=0.0, value=45.0, step=5.0
)

# --- Headline lift ---------------------------------------------------------
st.header("1. Is the campaign causing incremental conversions?")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Treatment conv. rate", f"{result.conv_rate_treatment:.3%}")
col2.metric("Control conv. rate", f"{result.conv_rate_control:.3%}")
col3.metric(
    "Absolute lift",
    f"{result.absolute_lift:.3%}",
    help=f"95% CI: [{result.ci_low:.3%}, {result.ci_high:.3%}]",
)
col4.metric("Relative lift", f"{result.relative_lift:.1%}")

sig_text = "statistically significant" if result.significant_at_05 else "not statistically significant"
st.write(
    f"The incremental lift is **{sig_text}** at the 5% level (p = {result.p_value:.2e}), "
    f"based on {result.n_treatment:,} treated and {result.n_control:,} control users. "
    f"This corresponds to **{result.incremental_conversions:,.0f} incremental conversions** "
    "in the treatment group -- conversions caused by the ad, not just correlated with it."
)

# --- Attribution vs incrementality -----------------------------------------
st.header("2. Attribution vs. incrementality")
naive_attributed = df.loc[df["treatment"] == 1, "conversion"].sum()
true_incremental = result.incremental_conversions

fig1, ax1 = plt.subplots(figsize=(5, 3.2))
bars = ax1.bar(
    ["Naive attribution\n(all treatment-group\nconversions)", "True incremental\n(caused by the ad)"],
    [naive_attributed, true_incremental],
    color=["#C44E52", "#4C72B0"],
)
for bar, val in zip(bars, [naive_attributed, true_incremental]):
    ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom")
ax1.set_ylabel("Conversions")

colA, colB = st.columns([1, 1])
with colA:
    st.pyplot(fig1)
with colB:
    naive_impact = compute_budget_impact(result.n_treatment, naive_attributed, cost_per_impression, revenue_per_conversion)
    true_impact = compute_budget_impact(result.n_treatment, true_incremental, cost_per_impression, revenue_per_conversion)
    overstatement = naive_attributed / true_incremental if true_incremental > 0 else float("inf")
    st.write(f"Naive attribution overstates causal impact by **{overstatement:.1f}x**.")
    comparison = pd.DataFrame(
        {
            "Naive (attribution)": {
                "Conversions counted": f"{naive_impact.incremental_conversions:,.0f}",
                "Incremental cost / conversion": f"${naive_impact.incremental_cost_per_conversion:,.2f}",
                "iROAS": f"{naive_impact.incremental_roas:.2f}x",
            },
            "True (incremental)": {
                "Conversions counted": f"{true_impact.incremental_conversions:,.0f}",
                "Incremental cost / conversion": f"${true_impact.incremental_cost_per_conversion:,.2f}",
                "iROAS": f"{true_impact.incremental_roas:.2f}x",
            },
        }
    )
    st.table(comparison)

# --- Uplift segments and targeting recommendation --------------------------
st.header("3. Which segments should get budget?")
st.caption("Users are ranked by a T-learner's predicted uplift and split into disjoint quintiles (Q5 = highest predicted uplift, Q1 = lowest).")

with st.spinner("Scoring uplift segments (cached after first run)..."):
    segments = get_uplift_segments(df)

fig2, ax2 = plt.subplots(figsize=(7, 3.2))
segments["estimated_incremental_lift"].plot(kind="bar", ax=ax2, color="#4C72B0")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Uplift segment")
ax2.set_ylabel("Observed incremental lift")
st.pyplot(fig2)
st.dataframe(segments)

best_segment = segments["estimated_incremental_lift"].idxmax()
worst_segment = segments["estimated_incremental_lift"].idxmin()
st.success(
    f"**Recommendation:** prioritize budget on segment **{best_segment}** "
    f"(highest observed incremental lift); deprioritize segment **{worst_segment}** -- those "
    "users convert (or don't) largely regardless of the ad, so spend there has little incremental payoff."
)

st.caption(
    "Assumptions & limitations: cost/revenue figures are illustrative; iROAS is for an "
    "intention-to-treat population, not per-impression; segment uplift estimates carry "
    "more sampling noise than the headline lift. See the project README for the full "
    "assumptions and limitations section."
)
