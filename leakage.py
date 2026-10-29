"""
leakage.py - condition 1: name the leaking columns and show the before/after.

A leaking column is one the hotel would not know at the moment the booking is
made, usually because it is filled in after the outcome. A model trained with
one scores beautifully in testing and then fails in real use.

The check: train the same quick logistic regression several times, each time
with a different set of columns, and compare the validation PR-AUC.
  - "all leaks in":     the honest features PLUS the leaking columns
  - "definite leaks out": reservation_status and its date removed
  - one row per suspect: the honest features plus that single suspect, to see
    whether it adds a suspiciously large jump

Owner: Bhooshan Sewock (draft prepared by the leader for the feedback session).

Run:  python leakage.py
"""

import config
from baseline import add_leak_features, quick_baseline
from split import get_split

# The honest feature set: things the hotel knows when the booking comes in.
# High-cardinality columns (country, agent, company) are left out of this
# quick check; they get embeddings in the real neural network.
HONEST_NUMERIC = [c for c in config.NUMERIC]
HONEST_CATEGORICAL = list(config.LOW_CARDINALITY_CATEGORICAL)


def run():
    df, labels = get_split()
    df = add_leak_features(df)
    # Rooms are reassigned at check-in, so "assigned differs from reserved"
    # mostly happens to guests who turned up. Logistic regression cannot build
    # this combination on its own, so we hand it over explicitly to expose it.
    df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    train, val = df[labels == "train"], df[labels == "val"]

    experiments = [
        ("All leaks in (status + status date + suspects + parking)",
         HONEST_NUMERIC + ["status_days", "booking_changes", "required_car_parking_spaces"],
         HONEST_CATEGORICAL + ["reservation_status", "assigned_room_type"]),
        ("Definite leaks out, suspects still in",
         HONEST_NUMERIC + ["booking_changes", "required_car_parking_spaces"],
         HONEST_CATEGORICAL + ["assigned_room_type"]),
        ("Clean: honest columns only",
         HONEST_NUMERIC, HONEST_CATEGORICAL),
        ("Clean + assigned_room_type only",
         HONEST_NUMERIC, HONEST_CATEGORICAL + ["assigned_room_type"]),
        ("Clean + room_changed (reserved != assigned)",
         HONEST_NUMERIC + ["room_changed"], HONEST_CATEGORICAL),
        ("Clean + booking_changes only",
         HONEST_NUMERIC + ["booking_changes"], HONEST_CATEGORICAL),
        ("Clean + required_car_parking_spaces only",
         HONEST_NUMERIC + ["required_car_parking_spaces"], HONEST_CATEGORICAL),
        ("Clean without days_in_waiting_list (kept: no effect)",
         [c for c in HONEST_NUMERIC if c != "days_in_waiting_list"], HONEST_CATEGORICAL),
    ]

    print("=" * 72)
    print("TEAM J - LEAKAGE CHECK (logistic regression, scored on VALIDATION)")
    print("=" * 72)
    print(f"{'Columns used':<52}{'PR-AUC':>9}{'ROC-AUC':>10}")
    results = {}
    for name, num, cat in experiments:
        r = quick_baseline(train, val, num, cat)
        results[name] = r
        print(f"{name:<52}{r['pr_auc']:>9.3f}{r['roc_auc']:>10.3f}")
    print("=" * 72)

    # Direct evidence for the suspects, from the whole dataset.
    changed = df["room_changed"] == 1
    print("Cancellation rate when the room was changed:  "
          f"{df.loc[changed, config.TARGET].mean():.1%}  ({changed.sum():,} bookings)")
    print("Cancellation rate when it was not:            "
          f"{df.loc[~changed, config.TARGET].mean():.1%}")
    parking = df["required_car_parking_spaces"] > 0
    print("Cancellation rate when parking was requested: "
          f"{df.loc[parking, config.TARGET].mean():.1%}  ({parking.sum():,} bookings)")
    zero = df["booking_changes"] == 0
    print("Cancellation rate with 0 booking changes:     "
          f"{df.loc[zero, config.TARGET].mean():.1%}")
    print("Cancellation rate with 1+ booking changes:    "
          f"{df.loc[~zero, config.TARGET].mean():.1%}")
    print("=" * 72)
    return results


if __name__ == "__main__":
    run()
