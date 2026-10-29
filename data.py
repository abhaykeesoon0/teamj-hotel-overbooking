"""
data.py - load the raw hotel bookings and run the quality checks.

What this module does, in order:
  1. Loads the raw CSV exactly as published (no rows removed).
  2. Builds two dates for every booking: the arrival date, and the booking
     date (arrival date minus lead_time). The booking date is needed for the
     split decision in condition 2 of the proposal feedback.
  3. Prints a quality report: size, missing values, exact duplicates, class
     balance, odd rows, and a leakage check.

What it deliberately does NOT do: drop duplicates, drop odd rows or drop
leaking columns. Each of those is a team decision that has to be argued in
the report with numbers, so this module only measures them. Cleaning happens
in a later step once the decisions are made.

Run on its own:  python data.py
"""

import numpy as np
import pandas as pd

import config


def load_raw(path=config.RAW_DATA_PATH):
    """Load the raw CSV and return it as a DataFrame with no rows removed.

    The file writes "NULL" for agent and company when there is none. pandas
    reads that as a missing value (NaN), which we keep for now. It does not
    mean the data is lost: NULL agent means the guest booked without a travel
    agent, and NULL company means it was not a company booking. features.py
    will turn these into their own category ("none") instead of dropping them.
    """
    df = pd.read_csv(path)
    return df


def add_dates(df):
    """Add arrival_date and booking_date columns to a copy of df.

    arrival_date is built from the three separate year/month/day columns.
    booking_date = arrival_date - lead_time days. The dataset has no booking
    date column, but lead_time is defined as the days between booking and
    arrival, so we can reconstruct it exactly.
    """
    df = df.copy()
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-"
        + df["arrival_date_month"] + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
    return df


def quality_report(df):
    """Print the data quality checks and return the key numbers as a dict.

    Every number printed here is meant to go straight into Section 2 (Data)
    of the report, so that the report and the code always agree.
    """
    n = len(df)
    results = {"rows": n, "columns": df.shape[1]}
    print("=" * 60)
    print("TEAM J - DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Source: {config.DATA_CITATION}\n")
    print(f"Rows: {n:,}   Columns: {df.shape[1]}")
    print(df["hotel"].value_counts().to_string(), "\n")

    # --- Date range -------------------------------------------------------
    print(f"Arrivals: {df['arrival_date'].min().date()} to {df['arrival_date'].max().date()}")
    print(f"Bookings made: {df['booking_date'].min().date()} to {df['booking_date'].max().date()}\n")

    # --- Missing values ---------------------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("Missing values (column: count, % of rows):")
    for col, count in missing.items():
        print(f"  {col:<12} {count:>7,}  ({count / n:.1%})")
    results["missing"] = missing.to_dict()
    print()

    # --- Target balance ---------------------------------------------------
    rate = df[config.TARGET].mean()
    print(f"Cancellation rate: {rate:.1%}  ({df[config.TARGET].sum():,} cancelled)")
    print(df.groupby("hotel")[config.TARGET].mean().map("{:.1%}".format).to_string(), "\n")
    results["cancel_rate"] = rate

    # --- Exact duplicates (condition 3, Arfi decides what to do) ----------
    # Checked on the original 32 columns only, so our two new date columns
    # cannot make two rows look different.
    original_cols = [c for c in df.columns if c not in ("arrival_date", "booking_date")]
    dup_mask = df.duplicated(subset=original_cols, keep="first")
    n_dup = int(dup_mask.sum())
    print(f"Exact duplicate rows (copies after the first): {n_dup:,} ({n_dup / n:.1%})")
    print(f"Rows left if duplicates were removed: {n - n_dup:,}")
    print(f"Cancellation rate among the duplicate copies: {df.loc[dup_mask, config.TARGET].mean():.1%}\n")
    results["duplicates"] = n_dup

    # --- Odd rows (flagged, not removed) ----------------------------------
    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    odd = {
        "zero guests": int((guests == 0).sum()),
        "zero nights": int((nights == 0).sum()),
        "negative adr (price)": int((df["adr"] < 0).sum()),
        "adr above 1000": int((df["adr"] > 1000).sum()),
    }
    print("Odd rows (flagged for review, nothing removed yet):")
    for name, count in odd.items():
        print(f"  {name:<22} {count:>5,}")
    results["odd_rows"] = odd
    print()

    # --- Leakage check (condition 1) --------------------------------------
    # If a column predicts the target perfectly, the model is reading the
    # answer, not learning anything. reservation_status is the clearest case.
    print("Leakage check: reservation_status vs is_canceled")
    print(pd.crosstab(df["reservation_status"], df[config.TARGET]).to_string())
    print("-> every row matches, so this column IS the answer and must be dropped.\n")

    # --- Cardinality (why we use embeddings) ------------------------------
    print("Distinct values in high-cardinality columns:")
    for col in config.HIGH_CARDINALITY_CATEGORICAL:
        print(f"  {col:<8} {df[col].nunique():>4}")
    print("=" * 60)
    return results


def load_data():
    """Main entry used by other modules: raw data plus the two date columns."""
    return add_dates(load_raw())


if __name__ == "__main__":
    np.random.seed(config.SEED)   # nothing random here yet, set for consistency
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    quality_report(data)
