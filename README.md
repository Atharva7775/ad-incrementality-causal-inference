# Incremental Ad Impact & Budget Allocation Using Causal Inference

A causal/incrementality measurement study on real randomized ad-experiment data —
built to demonstrate the difference between **"conversions after an ad"** (ordinary
marketing attribution) and **"conversions caused by an ad"** (incremental measurement),
and to show what that difference means for a budget decision.

## Business scenario

A performance marketing company runs display-ad campaigns through an ad platform. The
platform reports conversions among everyone who was shown (or eligible to be shown) an
ad — but some of those people would have converted anyway. This project answers:

1. Does advertising actually generate additional conversions?
2. How much incremental impact does the campaign create?
3. Which users respond to advertising, and which convert (or don't) regardless?
4. Should the company increase, reduce, or redirect its ad spend?

## Data

[Criteo Uplift Prediction Dataset (v2.1)](https://huggingface.co/datasets/criteo/criteo-uplift)
— real data from a randomized incrementality test (Diemert, Betlei, Renaudin, Amini,
*"A Large Scale Benchmark for Uplift Modeling"*, AdKDD 2018). A random share of users
were withheld from ad targeting (control); the rest were eligible to be targeted
(treatment). ~14M rows total; this project uses a reproducible ~10% random sample
(~1.4M rows, built by `src/data_loader.py`) to keep iteration fast.

Columns: `f0`–`f11` (anonymized features), `treatment`, `exposure`, `visit`, `conversion`.

> The dataset's official hosted mirrors (`go.criteo.net`, the `scikit-uplift` S3
> bucket) were unreachable at the time of writing; this project downloads from
> Criteo's own Hugging Face mirror instead — see "Setup" below.

## Key findings

*(from running the notebooks in this repo end-to-end on the 10% sample — see
`notebooks/` for full derivations)*

- **Causal lift (notebook 02):** Treatment conversion rate 0.320% vs. control 0.196% —
  an absolute lift of **0.123 percentage points** (95% CI: [0.102%, 0.145%]),
  a **62.7% relative lift**, highly significant (p < 1e-20). This corresponds to
  **~1,464 incremental conversions** in the treatment group.
- **Attribution vs. incrementality (notebook 03):** The ad platform's naive
  "attributed conversions" metric (all conversions in the treatment group, 3,799)
  **overstates the true causal impact by ~2.6x** relative to the incremental estimate
  (1,464). Under illustrative cost/revenue assumptions, this difference is the gap
  between an apparently-profitable campaign and its true (much lower) incremental ROAS.
- **Heterogeneous effects (notebook 04):** A causal forest (Qini AUC ≈0.82) finds a
  much sharper targeting signal than a T-learner baseline (Qini AUC ≈0.23): nearly all
  the incremental lift concentrates in one uplift quintile (≈1.4 percentage points),
  while the other four show no meaningful incremental effect.
- **Observational vs. experimental estimates (notebook 05):** A synthetic
  confounding exercise shows a naive comparison on non-randomized data can be badly
  biased (even flip sign), propensity-score weighting that models the confounder
  substantially corrects it, and propensity-score weighting that omits the confounder
  leaves meaningful bias — illustrating why observational inference is only as good
  as the confounders an analyst thinks to measure.

## Project structure

```
src/                 reusable modules (data loading, causal estimates, uplift models, cost assumptions)
notebooks/           01-05, the analysis in narrative form (see below)
dashboard/app.py     Streamlit dashboard: lift summary, attribution comparison, segment targeting, iROAS calculator
tests/               unit tests for the lift/CI/significance math
```

| Notebook | Covers |
|---|---|
| `01_data_overview_and_sql.ipynb` | DuckDB/SQL exploration, randomization/covariate-balance check |
| `02_causal_lift_estimation.ipynb` | Step 1: ITT lift, 95% CI, significance test |
| `03_attribution_vs_incrementality.ipynb` | Step 2: naive attribution vs. true incremental impact, business translation |
| `04_heterogeneous_uplift_modeling.ipynb` | Step 3: T-learner, causal forest, segment-level targeting recommendation |
| `05_observational_vs_experiment.ipynb` | Extension: propensity-score weighting vs. the RCT estimate, confounding bias demo |

## Setup

```bash
git clone <this-repo>
cd ad-incrementality-causal-inference
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the raw dataset (~297MB) into data/
mkdir -p data
curl -L -o data/criteo-research-uplift-v2.1.csv.gz \
  "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"

# Build the cached 10% sample used by every notebook/the dashboard
python3 src/data_loader.py

# Run the test suite
pytest tests/ -v

# Run the notebooks (or open them in Jupyter)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# Launch the dashboard
streamlit run dashboard/app.py
```

## Methodology notes

- **Primary estimate = intention-to-treat (ITT).** Because `treatment` is randomized
  (validated in notebook 01), the simple difference in conversion rates between
  treatment and control is an unbiased causal estimate — no regression adjustment is
  needed to make it valid, and adding one would only reduce variance, not fix bias.
- **Significance testing** uses a two-proportion z-test (pooled SE) cross-checked
  against an independent logistic-regression (HC1 robust SE) fit — see
  `src/causal_estimates.py` and `tests/test_causal_estimates.py` (the latter also
  cross-checks against `statsmodels.stats.proportion.proportions_ztest`).
- **Uplift/CATE modeling** compares a T-learner baseline against EconML's
  `CausalForestDML`; both are evaluated with the Qini AUC (`scikit-uplift`) and a
  disjoint-quintile segment table (rather than the library's default cumulative
  top-X% convention, which reads poorly as a "worst segment").

## Assumptions & limitations

- **Cost and revenue figures (`src/cost_assumptions.py`) are illustrative, not part
  of the dataset.** The Criteo data has no cost or revenue fields at all; iROAS and
  incremental-cost-per-conversion figures depend entirely on these assumptions and
  should be replaced with real campaign numbers before using this for an actual
  budget decision.
- **ITT, not per-protocol.** The effect estimated is of *eligibility* for targeting,
  not of actually seeing an ad (that would be the `exposure` column, a secondary
  analysis this project doesn't pursue). ITT is the right quantity for a budget
  decision about who to target, which is the decision this project is built around.
- **Causal forest results are fit on a 300k-row subsample** of the 1.4M-row sample
  (full-sample causal forest fitting was impractically slow for this project's scope)
  and should be re-validated on a larger or held-out sample before being used to
  actually reallocate budget.
- **The observational-comparison notebook's bias is synthetic** (deliberately
  constructed, since the real Criteo data is already randomized) to make the
  confounding mechanism concrete and verifiable. It illustrates the *mechanism* by
  which unobserved confounders bias observational estimates; it is not a claim about
  the size of such bias in any particular real non-randomized dataset.
- **Segment-level uplift estimates carry more sampling noise than the headline ITT
  estimate** — they're based on far fewer observations per segment and, for the
  causal forest, a smaller overall fit sample.
