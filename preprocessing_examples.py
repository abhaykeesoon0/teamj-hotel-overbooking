"""
preprocessing_examples.py - show the preprocessing on real bookings.

The brief asks the report to show preprocessing "before and after on real
examples". This script picks three real bookings that each exercise a
different decision, and prints them as they appear in the raw file and as
the models receive them. The table is also saved to
outputs/preprocessing_examples.csv so the report can quote it exactly.

The three bookings:
  A. no travel agent and no company (NULL in the file)
  B. a non-refundable deposit (the case where rules fail)
  C. a missing number of children

Steps shown:
  1. leaking columns removed (config.LEAKING_COLUMNS) and parking dropped
     as a precaution (config.PRECAUTIONARY_DROPS)
  2. missing categories become their own label instead of a dropped row
  3. numbers standardised with the training set's mean and spread only, so
     no information from validation or test leaks into the inputs

Owner: Nohur Muhammad Jamiil.

Run:  python preprocessing_examples.py
"""

import pandas as pd

import config
from split import get_split

SHOW_RAW = ["hotel", "lead_time", "adults", "children", "adr", "deposit_type",
            "agent", "company", "country", "reservation_status",
            "assigned_room_type", "booking_changes",
            "required_car_parking_spaces", config.TARGET]


def pick_examples(df):
    """Return one real booking for each case, chosen deterministically."""
    a = df[df["agent"].isna() & df["company"].isna()].index[0]
    b = df[df["deposit_type"] == "Non Refund"].index[0]
    c = df[df["children"].isna()].index[0]
    return {"A: no agent, no company": a, "B: non-refundable": b,
            "C: children missing": c}


def preprocess(df, labels):
    """Apply the team's preprocessing to every row (statistics from train)."""
    numeric, categorical = config.model_features()
    X = df[numeric + categorical].copy()
    for col in categorical:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
    train = labels == "train"
    for col in numeric:
        values = pd.to_numeric(X[col], errors="coerce")
        median = values[train].median()           # gap filler, from train only
        values = values.fillna(median)
        mean, std = values[train].mean(), values[train].std(ddof=0) or 1.0
        X[col] = ((values - mean) / std).round(2)
    return X


def run():
    df, labels = get_split()
    examples = pick_examples(df)
    after = preprocess(df, labels)

    dropped = config.LEAKING_COLUMNS + config.PRECAUTIONARY_DROPS
    print("=" * 70)
    print("PREPROCESSING ON REAL BOOKINGS (before and after)")
    print("=" * 70)
    print(f"Columns removed before modelling: {', '.join(dropped)}\n")

    rows = []
    for label, idx in examples.items():
        raw = df.loc[idx, SHOW_RAW]
        new = after.loc[idx]
        print(f"--- Booking {label} (row {idx}, {labels[idx]} set) ---")
        print("BEFORE:", raw.to_dict())
        print("AFTER (selected inputs):",
              new[["lead_time", "adults", "children", "adr", "deposit_type",
                   "agent", "company", "country"]].to_dict(), "\n")
        rows.append({"booking": label, "row_id": idx, "stage": "before",
                     **raw.astype(str).to_dict()})
        rows.append({"booking": label, "row_id": idx, "stage": "after",
                     **new.astype(str).to_dict()})

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "preprocessing_examples.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    run()
