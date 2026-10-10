"""Load the Criteo Uplift Prediction Dataset (v2.1) and cache a sampled parquet.

Source: https://huggingface.co/datasets/criteo/criteo-uplift (official Criteo mirror
of the v2.1 "unbiased" release used in Diemert et al., "A Large Scale Benchmark for
Uplift Modeling", AdKDD 2018). ~14M rows total; a randomized treatment assignment
makes it suitable for causal/incremental measurement, not just prediction.
"""

from pathlib import Path

import duckdb
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV_GZ = DATA_DIR / "criteo-research-uplift-v2.1.csv.gz"
SAMPLE_PARQUET = DATA_DIR / "criteo_sample.parquet"

FEATURE_COLS = [f"f{i}" for i in range(12)]


def build_sample(sample_frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Read the raw CSV, take a reproducible random sample, and cache it as parquet.

    Sampling is a simple uniform random sample (not stratified): treatment assignment
    in the raw data is itself randomized, so a uniform sample preserves that property
    and keeps the sampled treatment/control split representative of the full dataset.
    """
    if not RAW_CSV_GZ.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_CSV_GZ}. Download it from "
            "https://huggingface.co/datasets/criteo/criteo-uplift first."
        )

    df = pd.read_csv(RAW_CSV_GZ)
    sample = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PARQUET, index=False)
    return sample


def load_sample(sample_frac: float = 0.1, seed: int = 42, force_rebuild: bool = False) -> pd.DataFrame:
    """Return the cached sample, building it from the raw CSV if missing."""
    if force_rebuild or not SAMPLE_PARQUET.exists():
        return build_sample(sample_frac=sample_frac, seed=seed)
    return pd.read_parquet(SAMPLE_PARQUET)


def get_duckdb_connection(sample_frac: float = 0.1, seed: int = 42) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with a `criteo` view over the sample parquet."""
    load_sample(sample_frac=sample_frac, seed=seed)  # ensure parquet exists
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW criteo AS SELECT * FROM read_parquet('{SAMPLE_PARQUET.as_posix()}')")
    return con


if __name__ == "__main__":
    df = build_sample()
    print(f"Sample shape: {df.shape}")
    print(df["treatment"].value_counts(normalize=True))
