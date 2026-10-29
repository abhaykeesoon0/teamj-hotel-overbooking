"""
config.py - Team J shared settings.

Every other module imports its settings from here, so the seed, the file
paths and the column groups are defined in exactly one place. If a value
needs to change, change it here and nowhere else.

Owner: Nohur Muhammad Jamiil (team leader). The random seed and the split
settings belong to the leader, as required by condition 5 of the lecturer's
proposal feedback.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One seed for the whole project. Every module that uses randomness
# (numpy, scikit-learn, the neural network) must read it from here, so that
# running the script twice gives the same numbers in the report.
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Paths are built relative to this file rather than hardcoded, so the code
# runs on any teammate's machine without editing.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "hotels_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"      # figures and printed reports go here

# ---------------------------------------------------------------------------
# Data source (condition 4: provenance)
# ---------------------------------------------------------------------------
DATA_CITATION = (
    "Antonio, N., de Almeida, A. and Nunes, L. (2019) 'Hotel booking demand "
    "datasets', Data in Brief, 22, pp. 41-49. doi:10.1016/j.dib.2018.11.126. "
    "Licence: CC BY 4.0."
)
# Where our copy was downloaded from (combines the paper's H1 resort and
# H2 city files into one table with a "hotel" column):
DATA_DOWNLOAD_URL = (
    "https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11"
)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET = "is_canceled"   # 1 = booking was cancelled (includes no-shows), 0 = guest stayed

# ---------------------------------------------------------------------------
# Leakage (condition 1)
# ---------------------------------------------------------------------------
# Columns the hotel would not know at booking time. Evidence is printed by
# leakage.py; the before/after scores go in the report (condition 1).
LEAKING_COLUMNS = [
    "reservation_status",       # the outcome itself, written as text: matches
                                # is_canceled on every row
    "reservation_status_date",  # the date of the cancellation or check-out
    "assigned_room_type",       # given at check-in; when it differs from the
                                # reserved room only 5.4% cancel (vs 41.6%),
                                # because only guests who arrive get reassigned
    "booking_changes",          # final count of edits; a cancelled booking
                                # stops being edited, so the count is only
                                # known once the outcome is known
]

# Kept after checking: days_in_waiting_list is fixed once the booking is
# confirmed, and adding it to the baseline changed nothing (0.820 -> 0.820).
SUSPECTED_LEAKING_COLUMNS = []

# ---------------------------------------------------------------------------
# Columns used to build dates (not fed to the model as-is)
# ---------------------------------------------------------------------------
ARRIVAL_DATE_PARTS = [
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
]

# ---------------------------------------------------------------------------
# Precautionary drops (decided 26 Oct 2026, team leader)
# ---------------------------------------------------------------------------
# Not proven to leak, but too risky to keep. required_car_parking_spaces:
# not one of the 7,416 bookings that requested parking was cancelled. A
# perfect separation like this usually means the value is filled in at
# arrival (as with assigned_room_type), so a model would be reading the
# outcome. We drop it rather than risk a leak we cannot rule out.
PRECAUTIONARY_DROPS = ["required_car_parking_spaces"]

# ---------------------------------------------------------------------------
# FINAL feature list - every model and every analysis uses exactly these
# ---------------------------------------------------------------------------
# Rule for inclusion: the hotel knows the value when the booking arrives.
# High-cardinality categories: too many values for one-hot encoding to work
# well, which is why the neural network gives them learned embeddings.
HIGH_CARDINALITY_CATEGORICAL = ["country", "agent", "company"]

LOW_CARDINALITY_CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

NUMERIC = [
    "lead_time", "arrival_date_week_number",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "adr",
    "total_of_special_requests", "days_in_waiting_list",
]

# Country: the model is trained BOTH with and without it, and both results
# are reported. Portuguese bookings cancel at 56.6% against about 20% for
# other large markets, so the model would treat guests differently by
# nationality; the ethics section needs to show what that costs or gains.
# The main model uses INCLUDE_COUNTRY = True; the comparison run sets False.
INCLUDE_COUNTRY = True


def model_features(include_country=INCLUDE_COUNTRY):
    """Return (numeric, categorical) column lists for the models.

    One function so that every teammate gets the identical list, with the
    country switch applied the same way everywhere.
    """
    categorical = LOW_CARDINALITY_CATEGORICAL + [
        c for c in HIGH_CARDINALITY_CATEGORICAL
        if include_country or c != "country"
    ]
    return list(NUMERIC), categorical


# ---------------------------------------------------------------------------
# Split settings (condition 2)
# ---------------------------------------------------------------------------
# Which date the bookings are ordered by before splitting:
#   "arrival_date" - when the guest arrives
#   "booking_date" - when the booking was made (arrival_date minus lead_time)
# DECISION: arrival_date. Our decision layer works per night, and ordering by
# booking_date would scatter each night's bookings across train/val/test
# (test nights would hold as little as 11% of their bookings), and would skew
# the test set towards last-minute bookings. Arrival ordering allows a mild
# look-ahead; split.py --lookahead measures it so the report can show its size.
SPLIT_ORDER_COLUMN = "arrival_date"

# Share of bookings in each set, oldest first: train, then validation, then test.
# 70/15/15 keeps about 18,000 bookings in each of validation and test, enough
# to estimate PR-AUC and calibration reliably for a 37% minority class.
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# The split is saved here once and every module reads it from this file,
# so nobody ever creates their own split.
SPLIT_FILE = OUTPUT_DIR / "split.csv"
