"""
split.py - create the train / validation / test split ONCE, by time, and save it.

Why by time and not randomly: the hotel will use this model to predict future
bookings from past ones. A random split would let the model train on bookings
from after the ones it is tested on, which makes the score look better than it
would be in real use.

How it works:
  1. Sort every booking by the date chosen in config.SPLIT_ORDER_COLUMN
     (arrival_date or booking_date, still being decided).
  2. Find two cut-off dates so that roughly 70% of bookings fall before the
     first, 15% between the two, and 15% after the second.
  3. Label each booking train / val / test and save the labels to
     outputs/split.csv, keyed by the booking's row number in the raw CSV.

Every other module calls get_split() to read that saved file. The split is
never re-made anywhere else, which is how we guarantee every number in the
report comes from the same split.

Owner: Nohur Muhammad Jamiil (team leader), per condition 5.

Run on its own:  python split.py            (creates or re-uses the saved split)
                 python split.py --rebuild  (forces a new split, e.g. after
                                             changing SPLIT_ORDER_COLUMN)
                 python split.py --lookahead (also prints the evidence for the
                                             arrival-vs-booking decision)
"""

import sys

import pandas as pd

import config
from data import load_data


def make_split(df, order_col=config.SPLIT_ORDER_COLUMN):
    """Return a Series of 'train' / 'val' / 'test' labels, one per booking.

    We cut on whole dates rather than on an exact row count. If we cut by row
    count, bookings on the same day could land on both sides of the line,
    which would be a small time leak. Cutting by date keeps each day together.
    No randomness is involved, so the split is identical every time it runs.
    """
    dates = df[order_col]
    train_end = dates.quantile(config.TRAIN_FRACTION)
    val_end = dates.quantile(config.TRAIN_FRACTION + config.VAL_FRACTION)

    labels = pd.Series("test", index=df.index, name="set")
    labels[dates <= val_end] = "val"
    labels[dates <= train_end] = "train"
    return labels


def save_split(labels, order_col):
    """Save the labels with the booking row number and the ordering used."""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = labels.rename_axis("row_id").reset_index()
    out["ordered_by"] = order_col   # recorded so we can tell which split a file is
    out.to_csv(config.SPLIT_FILE, index=False)


def get_split(df=None, rebuild=False):
    """Main entry for every other module.

    Returns (df, labels): the full data and the matching train/val/test labels.
    If a saved split exists and was made with the current setting, it is
    re-used. Otherwise a new one is made and saved. Use it like this:

        df, labels = get_split()
        train = df[labels == "train"]
    """
    if df is None:
        df = load_data()
    order_col = config.SPLIT_ORDER_COLUMN

    if config.SPLIT_FILE.exists() and not rebuild:
        saved = pd.read_csv(config.SPLIT_FILE, index_col="row_id")
        # Refuse to silently use a split made under the other ordering.
        if saved["ordered_by"].iloc[0] == order_col and len(saved) == len(df):
            return df, saved["set"].reindex(df.index)
        print(f"Saved split was made by '{saved['ordered_by'].iloc[0]}', "
              f"config now says '{order_col}'. Rebuilding.")

    labels = make_split(df, order_col)
    save_split(labels, order_col)
    return df, labels


def split_report(df, labels):
    """Print the counts and a few checks for Section 2 of the report."""
    order_col = config.SPLIT_ORDER_COLUMN
    print("=" * 60)
    print(f"TEAM J - SPLIT REPORT (ordered by {order_col})")
    print("=" * 60)
    for name in ["train", "val", "test"]:
        part = df[labels == name]
        print(f"{name:<6} {len(part):>7,} bookings  "
              f"{part[order_col].min().date()} to {part[order_col].max().date()}  "
              f"cancel rate {part[config.TARGET].mean():.1%}  "
              f"median lead time {part['lead_time'].median():.0f} days")
    print(f"total  {len(labels):>7,}")

    # Duplicate check across sets (input for Arfi, condition 3). If an exact
    # copy of a test booking also sits in train, the model has effectively
    # seen that test booking already, which flatters the test score.
    original_cols = [c for c in df.columns if c not in ("arrival_date", "booking_date")]
    train_rows = set(map(tuple, df.loc[labels == "train", original_cols]
                         .astype(str).itertuples(index=False)))
    test_rows = df.loc[labels == "test", original_cols].astype(str).itertuples(index=False)
    overlap = sum(tuple(r) in train_rows for r in test_rows)
    print(f"\nTest bookings with an exact copy in train: {overlap:,}")
    print("=" * 60)


def lookahead_check(df, labels):
    """Measure the two costs behind the arrival-vs-booking decision (condition 2).

    1. Look-ahead under arrival ordering: some training bookings were MADE
       after some validation bookings were made. We train once on all of
       train, and once only on training bookings made before most validation
       bookings were made (no peeking). A random subset of the same size is
       the fair comparison, so the difference is due to timing, not size.
    2. Night completeness under booking ordering: for each arrival night in
       the latest 15% of dates, what share of that night's bookings would land
       in the test set if we ordered by booking_date instead.

    Scored on VALIDATION so the test set stays untouched.
    """
    import numpy as np
    from baseline import quick_baseline
    from leakage import HONEST_CATEGORICAL, HONEST_NUMERIC

    train, val = df[labels == "train"], df[labels == "val"]
    # Cut-off: the date by which 25% of validation bookings had been made.
    cutoff = val["booking_date"].quantile(0.25)
    no_peek = train[train["booking_date"] <= cutoff]
    rng = np.random.default_rng(config.SEED)
    same_size = train.iloc[rng.choice(len(train), len(no_peek), replace=False)]
    a = quick_baseline(no_peek, val, HONEST_NUMERIC, HONEST_CATEGORICAL)["pr_auc"]
    b = quick_baseline(same_size, val, HONEST_NUMERIC, HONEST_CATEGORICAL)["pr_auc"]
    print(f"PR-AUC (val), train only on bookings made by {cutoff.date()}: {a:.3f}  "
          f"({len(no_peek):,} rows)")
    print(f"PR-AUC (val), random training subset of the same size:     {b:.3f}")
    print(f"-> look-ahead effect: {b - a:+.3f}")

    # Night completeness if we had ordered by booking date instead.
    by_booking = make_split(df, "booking_date")
    late_nights = df["arrival_date"] >= df["arrival_date"].quantile(
        config.TRAIN_FRACTION + config.VAL_FRACTION)
    share = (by_booking[late_nights] == "test").groupby(df.loc[late_nights, "arrival_date"]).mean()
    print(f"\nIf ordered by booking_date, share of each late night's bookings in test: "
          f"median {share.median():.0%}, worst night {share.min():.0%}")
    print("=" * 60)


if __name__ == "__main__":
    data, split_labels = get_split(rebuild="--rebuild" in sys.argv)
    split_report(data, split_labels)
    if "--lookahead" in sys.argv:
        lookahead_check(data, split_labels)
    print(f"Split saved to {config.SPLIT_FILE}")
